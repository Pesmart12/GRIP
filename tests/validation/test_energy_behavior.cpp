#include <algorithm>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"

namespace grip {
namespace {

// Harmonic oscillator energy, exercised through the production
// step_body by recomputing u = -k*x each step from the
// current state -- no new force law added to src/. Gravity alone has
// no restoring force, so it has nothing periodic to bound-oscillate;
// this is the minimal system where the bounded-vs-drifting distinction
// actually exists. See docs/derivations/symplectic_euler.md.
double HarmonicEnergy(const RigidBodyState& state, double mass, double k) {
  return 0.5 * mass * state.v.x() * state.v.x() +
         0.5 * k * state.q.x() * state.q.x();
}

TEST(EnergyBehavior, SymplecticEulerBoundsOscillatorEnergy) {
  const double mass = 1.0;
  const double k = 1.0;
  const double omega = std::sqrt(k / mass);
  const double dt = 0.01;  // omega*dt = 0.01, well inside the |omega*dt|<2
                           // stability bound for this linear map.
  const double period = 2.0 * std::numbers::pi / omega;
  const int steps_per_period = static_cast<int>(period / dt);
  const int num_periods = 50;
  const int n = steps_per_period * num_periods;
  const int window = n / 10;

  RigidBodyState state;
  state.q.x() = 1.0;
  RigidBodyParams params;
  params.mass = mass;

  const double e0 = HarmonicEnergy(state, mass, k);
  double early_max_dev = 0.0;
  double late_max_dev = 0.0;

  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d u(-k * state.q.x(), 0.0, 0.0);
    state = step_body(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u, dt, /*gravity=*/0.0);

    const double dev = std::abs(HarmonicEnergy(state, mass, k) - e0);
    if (i < window) early_max_dev = std::max(early_max_dev, dev);
    if (i >= n - window) late_max_dev = std::max(late_max_dev, dev);
  }

  // Bounded oscillation, not secular drift: the energy excursion late
  // in a 50-period rollout should be the same order as early on, not
  // growing. Generous multiplicative slack for floating-point
  // accumulation over ~30k steps.
  EXPECT_LT(late_max_dev, 1.5 * early_max_dev);

  // The oscillation itself should be small relative to E0. Loose bound,
  // not tuned to the observed value -- the true amplitude scales with
  // omega*dt = 0.01, so 5% is generous headroom.
  EXPECT_LT(early_max_dev, 0.05 * e0);
  EXPECT_LT(late_max_dev, 0.05 * e0);
}

// Total energy of a body bouncing on a penalty spring: kinetic, plus
// gravitational potential referenced to the plane, plus the penalty
// potential U = sum_i (k/2) min(0, d_i)^2 that the spring stores while
// penetrating. The contact force is -grad U, so with no damping the whole
// system is conservative and this should stay put.
double BouncingEnergy(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, double gravity) {
  const double kinetic = 0.5 * params.mass * state.v.head<2>().squaredNorm() + 0.5 * params.inertia * state.v.z() * state.v.z();
  const double gravitational = params.mass * gravity * state.q.y();

  double stored = 0.0;
  for (const Contact& contact : detect_contacts_body(state, shape, plane)) {
    const double depth = std::min(0.0, contact.signed_distance);
    stored += 0.5 * penalty.stiffness * depth * depth;
  }
  return kinetic + gravitational + stored;
}

TEST(EnergyBehavior, UndampedPenaltyBounceBoundsEnergy) {
  // A unit square dropped flat onto the ground plane. With no damping the
  // contact is perfectly elastic -- the box rebounds to its drop height
  // and keeps going -- so energy should oscillate within a band rather
  // than drift, the same property the oscillator test above checks but
  // now through a real contact force rather than a stand-in wrench.
  //
  // Expect more jitter here than in the smooth oscillator: the backward
  // error analysis that gives symplectic integrators their bounded energy
  // assumes a smooth force, and the penalty force is C0 but not C1 at
  // d = 0. Volume preservation (det = 1) survives the kink because it is
  // algebra; energy conservation only nearly does.
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/1.0e4};
  const double gravity = 9.81;

  // omega = sqrt(2k/m) = 141 rad/s, so a contact episode lasts pi/omega
  // = 22 ms and dt resolves it with ~110 steps. That resolution, not the
  // |omega*dt| < 2 stability bound, is what caps k in practice.
  const double dt = 2.0e-4;
  const int n = 8000;  // about five bounces
  const int window = n / 10;

  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, 0.0);  // 0.1 m above first contact

  const double e0 = BouncingEnergy(state, params, shape, ground, penalty, gravity);
  double early_max_dev = 0.0;
  double late_max_dev = 0.0;
  double deepest = 0.0;

  for (int i = 0; i < n; ++i) {
    state = step_body(state, params, shape, ground, penalty, Eigen::Vector3d::Zero(), dt, gravity);

    const double dev = std::abs(BouncingEnergy(state, params, shape, ground, penalty, gravity) - e0);
    if (i < window) early_max_dev = std::max(early_max_dev, dev);
    if (i >= n - window) late_max_dev = std::max(late_max_dev, dev);
    deepest = std::min(deepest, state.q.y() - 0.5);
  }

  // The body actually bounced rather than resting or tunnelling.
  EXPECT_LT(deepest, -1.0e-3) << "never made contact";
  EXPECT_GT(deepest, -0.1) << "penetrated more than a tenth of the body";

  // Bounded, not secular: the late excursion is the same order as the
  // early one. Loose factor, because the kink makes per-bounce jitter
  // real and this test is about the absence of a trend, not its size.
  EXPECT_LT(late_max_dev, 3.0 * early_max_dev + 1.0e-9);
  EXPECT_LT(late_max_dev, 0.05 * e0);

  // Flat drop with a symmetric contact: nothing breaks the left/right
  // symmetry, so the box must never start rotating.
  EXPECT_DOUBLE_EQ(state.q.z(), 0.0);
  EXPECT_DOUBLE_EQ(state.v.z(), 0.0);
}

TEST(EnergyBehavior, DampedPenaltyContactDissipatesAndSettles) {
  // The one thing the undamped spring cannot do. With b > 0 the contact
  // removes energy on every bounce, so the box eventually stops bouncing
  // and comes to rest -- at the static equilibrium 2*k*(-d) = mg, i.e.
  // d* = -mg/(2k), where the damper contributes nothing because v = 0.
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0};
  const double gravity = 9.81;
  const double dt = 2.0e-4;
  const int n = 15000;  // 3 s, far longer than settling takes

  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 1.2, 0.0);

  const double e0 = BouncingEnergy(state, params, shape, ground, penalty, gravity);
  double highest_after_start = 0.0;

  for (int i = 0; i < n; ++i) {
    state = step_body(state, params, shape, ground, penalty, Eigen::Vector3d::Zero(), dt, gravity);
    highest_after_start = std::max(highest_after_start, BouncingEnergy(state, params, shape, ground, penalty, gravity));
  }

  // Energy never climbs above where it started. The contact can only
  // remove it, and gravity plus the spring are both conservative.
  EXPECT_LE(highest_after_start, e0 + 1.0e-9 * e0);

  // Came to rest, rather than merely slowing down.
  EXPECT_LT(state.v.norm(), 1.0e-6) << "still moving: v = " << state.v.transpose();

  // At the analytic resting penetration, not just somewhere near the
  // ground. This is the same fixed point test_penalty_force.cpp pins
  // exactly, reached here by simulation instead of by construction.
  const double resting_depth = -params.mass * gravity / (2.0 * penalty.stiffness);
  EXPECT_NEAR(state.q.y(), 0.5 + resting_depth, 1.0e-6);

  // Flat drop stays flat.
  EXPECT_DOUBLE_EQ(state.q.z(), 0.0);

  // A substantial fraction of the initial energy is gone, so this is
  // dissipation rather than a slow leak.
  const double e_final = BouncingEnergy(state, params, shape, ground, penalty, gravity);
  EXPECT_LT(e_final, 0.5 * e0);
}

TEST(EnergyBehavior, ExplicitEulerDriftsByContrast) {
  // Negative control: plain forward Euler on the same oscillator,
  // implemented locally here (not the production integrator), updates
  // position from the OLD velocity instead of the new one. That single
  // ordering change turns bounded oscillation into unbounded
  // exponential growth -- exactly what this suite must catch if the
  // update order in step_body ever regresses to this.
  const double mass = 1.0;
  const double k = 1.0;
  const double omega = std::sqrt(k / mass);
  const double dt = 0.01;
  const double period = 2.0 * std::numbers::pi / omega;
  const int steps_per_period = static_cast<int>(period / dt);
  const int num_periods = 50;
  const int n = steps_per_period * num_periods;

  double x = 1.0;
  double v = 0.0;
  const double e0 = 0.5 * mass * v * v + 0.5 * k * x * x;

  for (int i = 0; i < n; ++i) {
    const double x_old = x;
    const double v_old = v;
    v = v_old + dt * (-k * x_old / mass);
    x = x_old + dt * v_old;  // old velocity -- explicit, not semi-implicit
  }

  const double e_final = 0.5 * mass * v * v + 0.5 * k * x * x;
  EXPECT_GT(e_final, 5.0 * e0);
}

}  // namespace
}  // namespace grip
