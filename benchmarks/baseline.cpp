// Baseline throughput for GRIP 1.0, measured before the contact
// formulation changes underneath it.
//
// Two questions motivate this, and neither one is an optimization
// question:
//
//   1. The 2.0 comparison needs a 1.0 number. Penalty needs dt = 5e-4 to
//      resolve a bounce and an NCP solve runs at 1e-2, but that ratio is
//      arithmetic on paper until somebody measures what a step actually
//      costs. Once 2.0 is the default this measurement becomes
//      reconstructive rather than direct, so it is taken now.
//   2. Step 10 has to decide whether the Python boundary is crossed per
//      step or per batch, and that turns on how a step's cost compares
//      to a pybind11 round trip -- order 1 us. The us/step column below
//      is the input to that decision.
//
// Deliberately NOT measured: where the time goes inside a step. The four
// optimization questions in CLAUDE.md live mostly in code that 2.0
// replaces, and answering them now would be work with a known expiry.
//
// std::chrono and a calibration loop rather than a benchmark framework.
// "How many nanoseconds is a step" needs no statistical machinery, and a
// new dependency needs discussion first.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <Eigen/Core>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "gradient/rollout.hpp"

namespace {

using grip::BodyShape;
using grip::HalfPlane;
using grip::PenaltyParams;
using grip::RigidBodyParams;
using grip::RigidBodyState;
using grip::SystemControlVector;
using grip::SystemStateVector;

// The demo's constants, so these numbers describe a configuration that is
// already exercised by a running program rather than one invented here.
constexpr double kTimestep = 5.0e-4;
const PenaltyParams kPenalty{/*stiffness=*/1.0e4, /*damping=*/50.0, /*slip_damping=*/200.0, /*friction=*/0.5};

// Long enough that a damped contact episode is thoroughly over: the decay
// constant is about 20 ms at these constants, and this is 1000 steps.
constexpr int kSettleSteps = 1000;

// Horizon for the adjoint measurement. Short enough that a settled stack
// stays settled, long enough that the per-step figure is not dominated by
// the sweep's setup.
constexpr int kAdjointHorizon = 200;

constexpr double kCalibrationSeconds = 0.25;

// One simulated second at kTimestep, for the derived wall-clock column.
constexpr double kStepsPerSimulatedSecond = 1.0 / kTimestep;

// Written by every measurement and read only at the end: the accumulator
// exists so the optimizer cannot discard the work being measured.
double g_sink = 0.0;


BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}


struct Scene {
  const char* name;
  std::vector<RigidBodyState> states;
  std::vector<RigidBodyParams> params;
  std::vector<BodyShape> shapes;
};


std::vector<std::vector<Eigen::Vector3d>> ZeroControls(std::size_t bodies, int steps) {
  return std::vector<std::vector<Eigen::Vector3d>>(static_cast<std::size_t>(steps), std::vector<Eigen::Vector3d>(bodies, Eigen::Vector3d::Zero()));
}


// A tower of unit squares, dropped from 1 mm gaps and then simulated to
// rest. Settled numerically rather than placed at analytic equilibrium
// depths: the depth at level i carries the weight of every box above it,
// and reproducing that by hand is a derivation the benchmark does not
// need to own.
Scene MakeSettledStack(const char* name, std::size_t count) {
  Scene scene;
  scene.name = name;
  scene.params.assign(count, RigidBodyParams{/*mass=*/1.0, /*inertia=*/1.0 / 6.0});
  scene.shapes.assign(count, UnitSquare());
  scene.states.assign(count, RigidBodyState{});
  for (std::size_t i = 0; i < count; ++i) {
    scene.states[i].q = Eigen::Vector3d(0.0, 0.5 + static_cast<double>(i) * 1.001, 0.0);
  }

  const HalfPlane ground;
  const auto trajectory = grip::rollout_system(scene.states, scene.params, scene.shapes, ground, kPenalty, ZeroControls(count, kSettleSteps), kTimestep);
  scene.states = trajectory.back();
  return scene;
}


// A single box far above the plane. Every vertex is separated, so this
// times the path where detection reports contacts no force law will use:
// the cost of asking, with nothing to answer.
Scene MakeFreeFlight() {
  Scene scene;
  scene.name = "free flight, 1 body";
  scene.params.assign(1, RigidBodyParams{/*mass=*/1.0, /*inertia=*/1.0 / 6.0});
  scene.shapes.assign(1, UnitSquare());
  scene.states.assign(1, RigidBodyState{});
  scene.states[0].q = Eigen::Vector3d(0.0, 5.0, 0.0);
  return scene;
}


// Reported on every row so the table says what it measured rather than
// what it intended to measure. A stack that quietly separated while
// settling would otherwise be timing the free-flight path under a
// misleading name.
int CountActiveContacts(const Scene& scene) {
  const HalfPlane ground;
  int active = 0;
  for (std::size_t i = 0; i < scene.states.size(); ++i) {
    for (const grip::Contact& contact : grip::detect_contacts_body(scene.states[i], scene.shapes[i], ground)) {
      active += contact.signed_distance < 0.0 ? 1 : 0;
    }
    for (std::size_t j = i + 1; j < scene.states.size(); ++j) {
      for (const grip::PairContact& contact : grip::detect_contacts_pair(scene.states[i], scene.shapes[i], scene.states[j], scene.shapes[j])) {
        active += contact.signed_distance < 0.0 ? 1 : 0;
      }
    }
  }
  return active;
}


// Calls work() until the total is long enough to time reliably, then
// reports the per-call cost. Quadrupling rather than doubling, so the
// discarded calibration passes cost at most a third of the useful run.
template <typename Work>
double NanosecondsPerCall(const Work& work) {
  using Clock = std::chrono::steady_clock;
  std::size_t calls = 64;
  while (true) {
    const Clock::time_point start = Clock::now();
    for (std::size_t i = 0; i < calls; ++i) {
      g_sink += work();
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    if (elapsed > kCalibrationSeconds) {
      return 1.0e9 * elapsed / static_cast<double>(calls);
    }
    calls *= 4;
  }
}


// Every step is taken from the same settled state, so this is the cost of
// a step at a representative operating point rather than an average over
// a trajectory that drifts out of contact partway through.
double MeasureStep(const Scene& scene) {
  const HalfPlane ground;
  const std::vector<Eigen::Vector3d> controls(scene.states.size(), Eigen::Vector3d::Zero());
  return NanosecondsPerCall([&]() {
    return grip::step_system(scene.states, scene.params, scene.shapes, ground, kPenalty, controls, kTimestep)[0].q.y();
  });
}


// Per-step cost of the backward sweep, including the Jacobian the adjoint
// recomputes rather than tapes. Divided by the horizon so it is directly
// comparable to the forward column.
double MeasureAdjointStep(const Scene& scene) {
  const HalfPlane ground;
  const std::size_t bodies = scene.states.size();
  const auto controls = ZeroControls(bodies, kAdjointHorizon);
  const auto trajectory = grip::rollout_system(scene.states, scene.params, scene.shapes, ground, kPenalty, controls, kTimestep);

  const auto state_size = static_cast<Eigen::Index>(6 * bodies);
  const auto control_size = static_cast<Eigen::Index>(3 * bodies);
  std::vector<SystemStateVector> dl_dZ(static_cast<std::size_t>(kAdjointHorizon) + 1, SystemStateVector::Zero(state_size));
  dl_dZ.back() = SystemStateVector::Unit(state_size, 1);
  const std::vector<SystemControlVector> dl_dU(static_cast<std::size_t>(kAdjointHorizon), SystemControlVector::Zero(control_size));

  const double per_sweep = NanosecondsPerCall([&]() {
    return grip::adjoint_system(trajectory, scene.params, scene.shapes, ground, kPenalty, dl_dZ, dl_dU, kTimestep).dJ_dZ0(1);
  });
  return per_sweep / static_cast<double>(kAdjointHorizon);
}


void ReportScene(const Scene& scene) {
  const double forward = MeasureStep(scene);
  const double adjoint = MeasureAdjointStep(scene);
  std::printf("  %-22s %6d %11.0f %9.2f %12.0f %11.2f %11.2f\n", scene.name, CountActiveContacts(scene), forward, forward / 1000.0, 1.0e9 / forward, forward * kStepsPerSimulatedSecond / 1.0e9, adjoint / 1000.0);
}

}  // namespace

int main() {
#ifndef NDEBUG
  // Eigen without optimization is slower by more than an order of
  // magnitude, so a debug build does not produce a number worth writing
  // down. Loud rather than silent, because the two builds look identical
  // once the output is pasted somewhere.
  std::printf("\n  *** DEBUG BUILD -- these timings are meaningless. Configure with\n");
  std::printf("  *** -DCMAKE_BUILD_TYPE=Release and run that binary instead.\n\n");
#endif

  std::printf("GRIP baseline -- penalty contact, dt = %.0e, k = %.0e, b = %.0f, b_slip = %.0f, mu = %.1f\n\n", kTimestep, kPenalty.stiffness, kPenalty.damping, kPenalty.slip_damping, kPenalty.friction);
  std::printf("  %-22s %6s %11s %9s %12s %11s %11s\n", "scene", "active", "fwd ns/step", "fwd us/st", "steps/sec", "s per sim s", "adj us/st");
  std::printf("  %-22s %6s %11s %9s %12s %11s %11s\n", "----------------------", "------", "-----------", "---------", "------------", "-----------", "-----------");

  ReportScene(MakeFreeFlight());
  ReportScene(MakeSettledStack("resting pair, 2 bodies", 2));
  ReportScene(MakeSettledStack("stack, 10 bodies", 10));

  std::printf("\n  active      contacts with d < 0 -- plane vertices plus pair points\n");
  std::printf("  s per sim s wall-clock to simulate one second at dt = %.0e\n", kTimestep);
  std::printf("  adj us/st   one backward step, including the Jacobian it recomputes\n");
  std::printf("\n  (sink %.6f)\n", g_sink);
  return 0;
}
