// Convex polygons tumbling down a V-shaped pen, drawn with raylib.
//
// The flat notch at the bottom is the half-plane the whole project has
// been built against. The two ramps forming the V are ordinary bodies --
// given a large mass and their weight cancelled by a control wrench, so
// they stand in for immovable scenery without any special case in the
// physics. step_system takes a single HalfPlane, so anything beyond one
// flat floor has to be geometry, which is exactly what body-body contact
// is for.
//
// The ramps double as the walls rather than sitting inside separate ones:
// fewer heavy bodies pressing on each other, and no seams for a polygon
// to wedge into.
//
// Contact markers can be toggled on: every point detection reports, with
// its normal, in the direction the second body must move to escape. That
// is the clearest way to see what the library actually computes, as
// distinct from what the motion looks like.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <raylib.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"

namespace {

using grip::BodyShape;
using grip::HalfPlane;
using grip::PairContact;
using grip::PenaltyParams;
using grip::RigidBodyParams;
using grip::RigidBodyState;

constexpr int kScreenWidth = 1600;
constexpr int kScreenHeight = 1000;
constexpr double kWorldLeft = -4.0;
constexpr double kWorldBottom = -0.35;

// The horizontal span is what stays fixed: the window is resizable, and
// the scale is recomputed from its width each frame so the pen keeps
// filling it. Rendering only -- nothing here reaches the simulation, so
// the trajectory is identical at any window size.
constexpr double kWorldWidth = 8.0;
double g_pixels_per_metre = kScreenWidth / kWorldWidth;

constexpr double kGravity = 9.81;

// dt is set by contact resolution rather than stability. omega is about
// sqrt(k * Delassus) ~ 170 rad/s here, so an episode lasts ~18 ms and
// this resolves it with roughly forty steps. The stability bound would
// allow a far larger step and sample the bounce far too coarsely -- see
// docs/derivations/penalty_contact.md.
constexpr double kTimestep = 5.0e-4;
constexpr int kSubstepsPerFrame = 33;  // ~16.7 ms, so wall-clock real time

const PenaltyParams kDefaultPenalty{/*stiffness=*/1.0e4, /*damping=*/50.0, /*slip_damping=*/200.0, /*friction=*/0.5};

// A representative J M^-1 J^T for the polygons here, used only to put a
// number on the readout. The real one varies per contact with the moment
// arm; this is the right order of magnitude for a unit-ish body resting
// on a face.
constexpr double kNominalDelassus = 2.0;

constexpr double kStaticMass = 1.0e6;
constexpr std::size_t kStaticCount = 2;

// The pen is a V of two ramps, which double as its walls. Steeper than
// the friction angle atan(mu) = 26.6 degrees, so polygons slide down
// rather than sticking where they land -- below that angle this would be
// a very static demo, which is itself worth seeing by lowering it.
constexpr double kRampAngle = 0.62;  // ~35.5 degrees
constexpr double kRampHalfLength = 1.9;
constexpr double kRampHalfThickness = 0.15;

// The ramps stop short of each other, leaving a flat notch floored by the
// half-plane. Polygons collect there, resting on plane contact while
// leaning on body contact -- both paths visible at once.
constexpr double kApexHalfGap = 0.45;
constexpr double kApexHeight = 0.22;


// Hand-rolled rather than pulling in raygui: a slider is a track, a fill
// and a handle, and one more dependency for that is a poor trade.
//
// Stiffness spans four decades and is the only one worth a logarithmic
// track -- linear would bury the whole useful range in the first few
// pixels.
struct SliderSpec {
  const char* label;
  double* value;
  double lowest;
  double highest;
  bool logarithmic;
  const char* format;
};


constexpr int kSliderLeft = 16;
constexpr int kSliderWidth = 236;
constexpr int kSliderTop = 76;
constexpr int kSliderSpacing = 44;


double SliderFraction(const SliderSpec& slider) {
  if (slider.logarithmic) {
    return std::log(*slider.value / slider.lowest) / std::log(slider.highest / slider.lowest);
  }
  return (*slider.value - slider.lowest) / (slider.highest - slider.lowest);
}


// Returns the slider now being dragged, or -1. Dragging continues while
// the button is held even if the pointer wanders off the track, which is
// what makes a slider feel like a slider.
int UpdateSliders(const std::vector<SliderSpec>& sliders, int dragging) {
  const Vector2 mouse = GetMousePosition();
  if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
    dragging = -1;
  }
  for (std::size_t i = 0; i < sliders.size(); ++i) {
    const float track_y = static_cast<float>(kSliderTop + static_cast<int>(i) * kSliderSpacing + 22);
    const Rectangle grab{static_cast<float>(kSliderLeft - 8), track_y - 11.0f, static_cast<float>(kSliderWidth + 16), 24.0f};
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(mouse, grab)) {
      dragging = static_cast<int>(i);
    }
    if (dragging == static_cast<int>(i)) {
      const double t = std::clamp<double>((mouse.x - kSliderLeft) / kSliderWidth, 0.0, 1.0);
      const SliderSpec& slider = sliders[i];
      *slider.value = slider.logarithmic ? slider.lowest * std::pow(slider.highest / slider.lowest, t)
                                         : slider.lowest + t * (slider.highest - slider.lowest);
    }
  }
  return dragging;
}


void DrawSliders(const std::vector<SliderSpec>& sliders) {
  const Color track{58, 64, 78, 255};
  const Color fill{110, 190, 240, 255};
  const Color text{190, 200, 215, 255};

  for (std::size_t i = 0; i < sliders.size(); ++i) {
    const float track_y = static_cast<float>(kSliderTop + static_cast<int>(i) * kSliderSpacing + 22);
    const float filled = static_cast<float>(kSliderWidth * std::clamp(SliderFraction(sliders[i]), 0.0, 1.0));

    DrawText(TextFormat(sliders[i].format, sliders[i].label, *sliders[i].value), kSliderLeft, static_cast<int>(track_y) - 19, 16, text);
    DrawRectangle(kSliderLeft, static_cast<int>(track_y), kSliderWidth, 6, track);
    DrawRectangle(kSliderLeft, static_cast<int>(track_y), static_cast<int>(filled), 6, fill);
    DrawCircle(kSliderLeft + static_cast<int>(filled), static_cast<int>(track_y) + 3, 7.0f, fill);
  }
}


Vector2 ToScreen(const Eigen::Vector2d& world) {
  return Vector2{static_cast<float>((world.x() - kWorldLeft) * g_pixels_per_metre),
                 static_cast<float>(GetScreenHeight() - (world.y() - kWorldBottom) * g_pixels_per_metre)};
}


// Not called Rectangle: raylib already has a type by that name, and
// shadowing it from an anonymous namespace compiles but reads badly.
BodyShape BoxShape(double half_width, double half_height) {
  return BodyShape{{{-half_width, -half_height}, {half_width, -half_height}, {half_width, half_height}, {-half_width, half_height}}};
}


// Counterclockwise by construction, which is what detection requires --
// the outward normal of an edge is its direction rotated by -90 degrees,
// and that only points outward for CCW winding.
BodyShape RegularPolygon(int sides, double radius) {
  BodyShape shape;
  for (int i = 0; i < sides; ++i) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / sides;
    shape.vertices.emplace_back(radius * std::cos(angle), radius * std::sin(angle));
  }
  return shape;
}


// Area and second moment of a regular n-gon of circumradius r, at unit
// density. I/m = (r^2/6)(1 + 2cos^2(pi/n)), which reduces to r^2/3 for a
// square -- the familiar a^2/6 with a = r*sqrt(2).
RigidBodyParams PolygonParams(int sides, double radius) {
  const double area = 0.5 * sides * radius * radius * std::sin(2.0 * std::numbers::pi / sides);
  const double cosine = std::cos(std::numbers::pi / sides);
  return RigidBodyParams{area, area * radius * radius * (1.0 + 2.0 * cosine * cosine) / 6.0};
}


struct Scene {
  std::vector<RigidBodyState> states;
  std::vector<RigidBodyParams> params;
  std::vector<BodyShape> shapes;
  std::vector<Color> colors;
  std::mt19937 generator{20260813};  // fixed, so a reset reproduces exactly
};


void AddPolygon(Scene* scene, double x, double y) {
  std::uniform_int_distribution<int> sides(3, 6);
  std::uniform_real_distribution<double> radius(0.16, 0.30);
  std::uniform_real_distribution<double> spin(-4.0, 4.0);
  std::uniform_real_distribution<double> drift(-1.5, 1.5);

  const int count = sides(scene->generator);
  const double size = radius(scene->generator);

  RigidBodyState state;
  state.q = Eigen::Vector3d(x, y, drift(scene->generator));
  state.v = Eigen::Vector3d(drift(scene->generator), 0.0, spin(scene->generator));

  scene->states.push_back(state);
  scene->params.push_back(PolygonParams(count, size));
  scene->shapes.push_back(RegularPolygon(count, size));
  scene->colors.push_back(ColorFromHSV(static_cast<float>(scene->states.size() * 47 % 360), 0.65f, 0.95f));
}


Scene BuildScene() {
  Scene scene;

  // Each ramp is placed by its lower end, then extended back up along its
  // own axis -- so the apex gap and the slope are the two numbers that
  // describe the pen, rather than a pair of centres to keep in sync.
  for (const double side : {-1.0, 1.0}) {
    // Rotating by +angle lifts a body's right-hand end, so the left ramp
    // takes a negative angle to fall towards the centre and the right
    // ramp a positive one. Its lower end is then the -along extreme on
    // the right and the +along extreme on the left, which is the same
    // sign again.
    const double angle = side * kRampAngle;
    const Eigen::Vector2d along(std::cos(angle), std::sin(angle));
    const Eigen::Vector2d lower_end(side * kApexHalfGap, kApexHeight);

    RigidBodyState ramp;
    const Eigen::Vector2d centre = lower_end + side * kRampHalfLength * along;
    ramp.q = Eigen::Vector3d(centre.x(), centre.y(), angle);

    scene.states.push_back(ramp);
    scene.params.push_back(RigidBodyParams{kStaticMass, kStaticMass});
    scene.shapes.push_back(BoxShape(kRampHalfLength, kRampHalfThickness));
    scene.colors.push_back(Color{90, 96, 110, 255});
  }

  // One to start with; A adds more.
  std::uniform_real_distribution<double> across(-1.4, 1.4);
  std::uniform_real_distribution<double> height(2.2, 3.4);
  for (int i = 0; i < 1; ++i) {
    AddPolygon(&scene, across(scene.generator), height(scene.generator));
  }
  return scene;
}


// Cancel gravity on the ramps only. Everything else falls.
std::vector<Eigen::Vector3d> Controls(const Scene& scene) {
  std::vector<Eigen::Vector3d> u(scene.states.size(), Eigen::Vector3d::Zero());
  for (std::size_t i = 0; i < kStaticCount; ++i) {
    u[i] = Eigen::Vector3d(0.0, scene.params[i].mass * kGravity, 0.0);
  }
  return u;
}


std::vector<Eigen::Vector2d> WorldVertices(const RigidBodyState& state, const BodyShape& shape) {
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();
  std::vector<Eigen::Vector2d> world(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    world[i] = state.q.head<2>() + rotation * shape.vertices[i];
  }
  return world;
}


void DrawBody(const RigidBodyState& state, const BodyShape& shape, Color color) {
  const std::vector<Eigen::Vector2d> corners = WorldVertices(state, shape);
  for (std::size_t i = 0; i < corners.size(); ++i) {
    DrawLineEx(ToScreen(corners[i]), ToScreen(corners[(i + 1) % corners.size()]), 2.5f, color);
  }
  // A spoke to the first vertex, so rotation is visible on a shape whose
  // outline is nearly rotationally symmetric.
  DrawLineEx(ToScreen(state.q.head<2>()), ToScreen(corners.front()), 1.0f, Fade(color, 0.45f));
}


void DrawContacts(const Scene& scene, const HalfPlane& floor) {
  const Color marker{255, 210, 60, 255};

  for (std::size_t i = 0; i < scene.states.size(); ++i) {
    for (const grip::Contact& contact : grip::detect_contacts_body(scene.states[i], scene.shapes[i], floor)) {
      if (contact.signed_distance >= 0.0) {
        continue;
      }
      DrawCircleV(ToScreen(contact.point), 4.0f, marker);
      DrawLineEx(ToScreen(contact.point), ToScreen(contact.point + 0.22 * contact.normal), 2.0f, marker);
    }
    for (std::size_t j = i + 1; j < scene.states.size(); ++j) {
      for (const PairContact& contact : grip::detect_contacts_pair(scene.states[i], scene.shapes[i], scene.states[j], scene.shapes[j])) {
        if (contact.signed_distance >= 0.0) {
          continue;
        }
        DrawCircleV(ToScreen(contact.point), 4.0f, marker);
        DrawLineEx(ToScreen(contact.point), ToScreen(contact.point + 0.22 * contact.normal), 2.0f, marker);
      }
    }
  }
}

}  // namespace

int main() {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(kScreenWidth, kScreenHeight, "GRIP - polygons in a pen");
  SetTargetFPS(60);

  const HalfPlane floor;  // y >= 0
  Scene scene = BuildScene();
  PenaltyParams penalty = kDefaultPenalty;
  bool paused = false;
  bool show_contacts = true;
  int dragging = -1;

  const std::vector<SliderSpec> sliders = {
      {"stiffness  k", &penalty.stiffness, 1.0e2, 1.0e6, true, "%s  %.0f N/m"},
      {"damping  b", &penalty.damping, 0.0, 300.0, false, "%s  %.0f N.s/m"},
      {"slip damping  b_slip", &penalty.slip_damping, 0.0, 1000.0, false, "%s  %.0f N.s/m"},
      {"friction  mu", &penalty.friction, 0.0, 1.5, false, "%s  %.2f"},
  };

  while (!WindowShouldClose()) {
    g_pixels_per_metre = GetScreenWidth() / kWorldWidth;
    dragging = UpdateSliders(sliders, dragging);

    if (IsKeyPressed(KEY_SPACE)) {
      paused = !paused;
    }
    if (IsKeyPressed(KEY_R)) {
      scene = BuildScene();
    }
    if (IsKeyPressed(KEY_C)) {
      show_contacts = !show_contacts;
    }
    if (IsKeyPressed(KEY_A)) {
      AddPolygon(&scene, 0.0, 3.6);
    }

    if (!paused) {
      const std::vector<Eigen::Vector3d> u = Controls(scene);
      for (int step = 0; step < kSubstepsPerFrame; ++step) {
        scene.states = grip::step_system(scene.states, scene.params, scene.shapes, floor, penalty, u, kTimestep, kGravity);
      }
    }

    BeginDrawing();
    ClearBackground(Color{18, 20, 26, 255});

    DrawLineEx(ToScreen({kWorldLeft, 0.0}), ToScreen({-kWorldLeft, 0.0}), 3.0f, Color{120, 130, 145, 255});
    for (std::size_t i = 0; i < scene.states.size(); ++i) {
      DrawBody(scene.states[i], scene.shapes[i], scene.colors[i]);
    }
    if (show_contacts) {
      DrawContacts(scene, floor);
    }

    DrawRectangle(8, 62, 252, kSliderSpacing * static_cast<int>(sliders.size()) + 54, Color{12, 14, 20, 205});
    DrawSliders(sliders);

    // The number that actually governs whether a bounce is integrated or
    // merely stepped over. Stability would allow k far past the point
    // where this collapses, which is why resolution and not stability is
    // what caps stiffness in practice -- see
    // docs/derivations/penalty_contact.md.
    const double steps_in_contact = std::numbers::pi / (kTimestep * std::sqrt(penalty.stiffness * kNominalDelassus));
    const Color verdict = steps_in_contact > 15.0  ? Color{120, 220, 150, 255}
                          : steps_in_contact > 7.0 ? Color{255, 210, 60, 255}
                                                   : Color{240, 110, 100, 255};
    DrawText(TextFormat("~%.0f steps inside a bounce", steps_in_contact), kSliderLeft, 62 + kSliderSpacing * static_cast<int>(sliders.size()) + 24, 17, verdict);

    DrawText(TextFormat("%zu bodies   %d substeps @ dt=%.0e   %d fps", scene.states.size(), kSubstepsPerFrame, kTimestep, GetFPS()), 14, 12, 18, Color{170, 180, 195, 255});
    DrawText("SPACE pause    A add    C contacts    R reset", 14, 36, 18, Color{110, 120, 135, 255});
    DrawText(paused ? "PAUSED" : "", GetScreenWidth() - 100, 12, 18, Color{255, 210, 60, 255});

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
