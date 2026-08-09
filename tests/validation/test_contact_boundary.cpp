#include <cmath>
#include <functional>
#include <limits>

#include <gtest/gtest.h>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

// Behaviour at the activation boundary is recorded here rather than held
// to a tolerance. CLAUDE.md: near contact activation, gradients are
// *expected* to misbehave, and characterizing how is the point.
//
// The file is a comparison between two severity classes. With a spring
// alone the contact force is continuous but not differentiable at d = 0,
// so a central difference across the boundary is wrong but **bounded**.
// Adding a damper makes the force itself discontinuous on entry, and the
// same sweep diverges like 1/eps. Which class a formulation lands in is
// a measurement, and this is the measurement. The tests above the
// kDamping constant use b = 0; the Damped* ones below it turn the damper
// on at the same operating point.

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

Eigen::Matrix3d FiniteDifferenceForceJacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, double eps) {
  const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> force_of_q = [&](const Eigen::Vector3d& q) {
    RigidBodyState perturbed = state;
    perturbed.q = q;
    return penalty_force_body(perturbed, shape, plane, penalty);
  };
  return testutil::CentralDifferenceJacobian<3, 3>(force_of_q, state.q, eps);
}

// A square resting flat with both bottom corners exactly on the plane:
// every quantity here is a dyadic rational, so d = 0 holds bit-exactly
// rather than to within rounding.
RigidBodyState OnTheBoundary() {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.5, 0.0);
  return state;
}

constexpr double kStiffness = 128.0;

// Approaching from inside, both bottom corners are active and each
// contributes -k to d(f_y)/d(c_y). From outside, neither does. The
// derivative genuinely does not exist at d = 0; these are the two
// one-sided limits it falls between.
constexpr double kFromInside = -2.0 * kStiffness;
constexpr double kFromOutside = 0.0;
constexpr double kMidpoint = 0.5 * (kFromInside + kFromOutside);

// Floating-point floor on a central difference taken at this operating
// point, independent of the kink. Perturbing q_y = 0.5 by eps rounds, so
// the step actually taken differs from the nominal one by up to half a
// ULP of 0.5; the estimate inherits that as a relative error and the
// stiffness amplifies it. This is the O(machine_eps/eps) cancellation
// term documented in tests/utils/finite_difference.hpp, which is why
// that helper defaults to eps ~ 6e-6 rather than something smaller.
double CancellationFloor(double eps) {
  constexpr double kOperatingPoint = 0.5;
  return std::abs(kMidpoint) * std::numeric_limits<double>::epsilon() * kOperatingPoint / (2.0 * eps);
}

TEST(ContactBoundary, CentralDifferenceReturnsTheMidpointOfTheOneSidedSlopes) {
  const RigidBodyState state = OnTheBoundary();
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness};

  // Swept over the range where a central difference is actually
  // meaningful. The estimate lands on the average of the two one-sided
  // slopes -- matching neither branch, which is the whole finding.
  for (const double eps : {1.0e-2, 1.0e-4, 1.0e-6}) {
    const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, ground, penalty, eps);
    EXPECT_NEAR(fd(1, 1), kMidpoint, CancellationFloor(eps) + std::abs(kMidpoint) * 1.0e-12) << "eps = " << eps;
  }
}

TEST(ContactBoundary, FiniteDifferenceStaysBoundedAsTheStepShrinks) {
  // The property that separates a kink from a jump, and the reason this
  // file exists. A central difference straddling a kink returns a finite
  // wrong number no matter how small the step gets. Across a *jump* --
  // which is what adding a damper produces -- the same quantity grows
  // like 1/eps without bound. Ten orders of magnitude in eps must not
  // move this estimate beyond the cancellation floor.
  const RigidBodyState state = OnTheBoundary();
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness};

  for (const double eps : {1.0e-1, 1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9, 1.0e-11}) {
    const double estimate = FiniteDifferenceForceJacobian(state, shape, ground, penalty, eps)(1, 1);

    // Bounded strictly between the two one-sided slopes, for every eps.
    EXPECT_LE(std::abs(estimate), std::abs(kFromInside)) << "eps = " << eps;
    EXPECT_NEAR(estimate, kMidpoint, CancellationFloor(eps) + std::abs(kMidpoint) * 1.0e-12) << "eps = " << eps;
  }
}

TEST(ContactBoundary, AnalyticAndFiniteDifferenceDisagreeByExactlyOneSidedHalf) {
  // Recording the size of the discrepancy, not asserting it away. At
  // d = 0 the implementation reports the separated branch (matching the
  // force value, which is zero there), so the analytic answer is 0 while
  // the central difference is the midpoint -k. The gap is exactly half
  // the one-sided jump, and the jump itself is the stiffness.
  const RigidBodyState state = OnTheBoundary();
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness};

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, ground, penalty, 1.0e-6);

  EXPECT_TRUE(analytic.df_dq.isZero(0.0)) << "at d = 0 the separated branch is reported";
  EXPECT_NEAR(fd(1, 1) - analytic.df_dq(1, 1), -kStiffness, kStiffness * 1.0e-9);
}

TEST(ContactBoundary, AwayFromTheBoundaryTheyAgree) {
  // The disagreement above is a property of the boundary, not a bug in
  // the Jacobian: a step's width clear of d = 0 on either side, analytic
  // and finite difference match to ordinary FD accuracy.
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness};

  for (const double offset : {-1.0e-2, 1.0e-2}) {  // penetrating, then separated
    RigidBodyState state = OnTheBoundary();
    state.q.y() += offset;

    const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
    const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, ground, penalty, 1.0e-6);

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(analytic.df_dq(i, j), fd(i, j), 1.0e-6) << "offset " << offset << " at (" << i << ", " << j << ")";
      }
    }
  }
}

TEST(ContactBoundary, GeometricStiffnessVanishesIntoTheBoundary) {
  // The jump in df_c/dq is entirely the material term. The geometric term
  // is proportional to lambda_i, which goes to zero with the penetration
  // depth, so it approaches the boundary continuously from inside.
  //
  // At theta = 0 both bottom corners have arm = (+-0.5, -0.5), so
  // n . arm = -0.5 and J_z = -+0.5 for each. The material part of the
  // (theta, theta) entry is -k*(0.5^2 + 0.5^2) = -k/2. Each corner's
  // geometric part is -lambda*(n . arm) = +lambda/2, so the two together
  // contribute lambda = k*depth.
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness};
  const double material = -0.5 * kStiffness;

  // Dyadic depths, so 0.5 - depth is exactly representable and the whole
  // chain -- vertex position, d, lambda, both stiffness terms -- is exact
  // in binary. No tolerance needed to make the point.
  for (const double depth : {1.0 / 128.0, 1.0 / 16384.0, 1.0 / 2097152.0}) {
    RigidBodyState state = OnTheBoundary();
    state.q.y() -= depth;

    const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
    const double geometric = analytic.df_dq(2, 2) - material;

    EXPECT_DOUBLE_EQ(geometric, kStiffness * depth) << "depth " << depth;
    // Shrinking with the depth, while the material term does not: the
    // jump at the boundary is the material term and nothing else.
    EXPECT_LT(std::abs(geometric), std::abs(material)) << "depth " << depth;
  }
}


constexpr double kDamping = 50.0;
constexpr double kClosingSpeed = 1.0;

// Same box on the boundary, but arriving at 1 m/s so ddot = -1 at both
// bottom corners.
RigidBodyState ArrivingAtTheBoundary() {
  RigidBodyState state = OnTheBoundary();
  state.v = Eigen::Vector3d(0.0, -kClosingSpeed, 0.0);
  return state;
}

TEST(ContactBoundary, DampedForceJumpsOnEntry) {
  // The spring term dies smoothly into the boundary; the damping term
  // does not. Approaching from inside, lambda tends to b*|ddot| rather
  // than to zero, so the force itself is discontinuous -- a step, not a
  // kink. The clamp does not help here: on entry ddot < 0, so
  // -k*d - b*ddot is strictly positive and the max never engages.
  const RigidBodyState state = ArrivingAtTheBoundary();
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness, kDamping};

  // Two active corners, each tending to b*|ddot|.
  const double jump = 2.0 * kDamping * kClosingSpeed;

  for (const double depth : {1.0e-4, 1.0e-8, 1.0e-12}) {
    RigidBodyState just_inside = state;
    just_inside.q.y() -= depth;
    RigidBodyState just_outside = state;
    just_outside.q.y() += depth;

    const double inside = penalty_force_body(just_inside, shape, ground, penalty).y();
    const double outside = penalty_force_body(just_outside, shape, ground, penalty).y();

    EXPECT_TRUE(penalty_force_body(just_outside, shape, ground, penalty).isZero(0.0));
    EXPECT_NEAR(inside - outside, jump, jump * 1.0e-6 + 2.0 * kStiffness * depth) << "depth " << depth;
  }
}

TEST(ContactBoundary, DampedFiniteDifferenceDivergesAsTheStepShrinks) {
  // The measurement that separates the two severity classes. Straddling
  // a jump of size 2*b*|ddot|, a central difference returns
  //
  //   (0 - (2*k*eps + 2*b*|ddot|)) / (2*eps) = -k - b*|ddot|/eps
  //
  // which grows without bound as eps shrinks -- roughly a factor of ten
  // per decade. The undamped sweep above sits still at -k instead.
  const RigidBodyState state = ArrivingAtTheBoundary();
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness, kDamping};

  double previous = 0.0;
  for (const double eps : {1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5}) {
    const double estimate = FiniteDifferenceForceJacobian(state, shape, ground, penalty, eps)(1, 1);
    const double predicted = -kStiffness - kDamping * kClosingSpeed / eps;

    EXPECT_NEAR(estimate, predicted, std::abs(predicted) * 1.0e-6) << "eps = " << eps;
    // Blows past the undamped case's bound, which was 2*k.
    EXPECT_GT(std::abs(estimate), 2.0 * kStiffness) << "eps = " << eps;
    if (previous != 0.0) {
      EXPECT_GT(std::abs(estimate) / previous, 5.0) << "eps = " << eps << " did not grow like 1/eps";
    }
    previous = std::abs(estimate);
  }
}

TEST(ContactBoundary, ClampMakesTheExitContinuousButNotTheEntry) {
  // The clamp is asymmetric, and it is worth being precise about which
  // half it fixes.
  //
  // Leaving (ddot > 0), it engages while the vertex is still penetrating
  // -- as soon as b*ddot exceeds -k*d -- so lambda reaches zero strictly
  // before d does and separation is continuous.
  //
  // Entering (ddot < 0), -k*d - b*ddot is positive the instant d goes
  // negative, so the max never engages and the force steps.
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{kStiffness, kDamping};

  // Departing at 1 m/s, the clamp binds once k*depth < b*1, i.e. for any
  // depth below b/k = 0.39 m -- which is every depth that matters.
  for (const double depth : {1.0e-2, 1.0e-4, 1.0e-8}) {
    RigidBodyState leaving = OnTheBoundary();
    leaving.q.y() -= depth;
    leaving.v = Eigen::Vector3d(0.0, kClosingSpeed, 0.0);  // moving away
    EXPECT_TRUE(penalty_force_body(leaving, shape, ground, penalty).isZero(0.0)) << "adhesion at depth " << depth;
  }

  // Arriving at the same speeds and depths, the force is already at full
  // damping strength.
  RigidBodyState arriving = OnTheBoundary();
  arriving.q.y() -= 1.0e-8;
  arriving.v = Eigen::Vector3d(0.0, -kClosingSpeed, 0.0);
  EXPECT_NEAR(penalty_force_body(arriving, shape, ground, penalty).y(), 2.0 * kDamping * kClosingSpeed, 1.0e-3);
}

}  // namespace
}  // namespace grip
