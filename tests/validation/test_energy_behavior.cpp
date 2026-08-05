#include <algorithm>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

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
    state = step_body(state, params, u, dt, /*gravity=*/0.0);

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
