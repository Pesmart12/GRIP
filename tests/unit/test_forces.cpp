#include <gtest/gtest.h>

#include <Eigen/Core>

#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {
namespace {

TEST(GravityForce, ActsThroughTheCenterOfMassWithNoTorque) {
  const RigidBodyParams params{/*mass=*/2.5, /*inertia=*/0.4};

  const Eigen::Vector3d force = gravity_force(params, 9.81);

  // y-up: gravity is -y, scaled by mass. It acts through the COM, so the
  // torque component is an identical zero rather than something small.
  EXPECT_DOUBLE_EQ(force.x(), 0.0);
  EXPECT_DOUBLE_EQ(force.y(), -2.5 * 9.81);
  EXPECT_DOUBLE_EQ(force.z(), 0.0);
}

TEST(GravityForce, ScalesWithGravityAndVanishesAtZero) {
  const RigidBodyParams params{/*mass=*/2.5, /*inertia=*/0.4};

  EXPECT_DOUBLE_EQ(gravity_force(params, 0.0).y(), 0.0);
  EXPECT_DOUBLE_EQ(gravity_force(params, 2.0 * 9.81).y(), 2.0 * gravity_force(params, 9.81).y());
}

TEST(GravityForce, JacobianIsExactlyZero) {
  // Gravity is a constant wrench: independent of q, v, and everything else.
  // Both blocks must come out as identical zeros, not merely small.
  //
  // This lives here rather than in the integrator tests because that is what
  // it actually checks. Feeding a zero ForceJacobian to the integrator and
  // observing zeros back tests nothing; whether gravity_force_jacobian
  // returns zeros is the substantive claim.
  const RigidBodyParams params{/*mass=*/2.5, /*inertia=*/0.4};
  const Eigen::Vector3d q(1.0, 2.0, 3.0);
  const Eigen::Vector3d v(-1.0, 0.5, -0.2);

  const ForceJacobian jac = gravity_force_jacobian(q, v, params, kDefaultGravity);

  EXPECT_TRUE(jac.df_dq.isZero(0.0));
  EXPECT_TRUE(jac.df_dv.isZero(0.0));
}

TEST(ForceJacobian, DefaultsToZeroInBothBlocks) {
  // The assembly layer sums ForceJacobians starting from a default-
  // constructed one, so "default" has to mean "contributes nothing".
  const ForceJacobian jac;

  EXPECT_TRUE(jac.df_dq.isZero(0.0));
  EXPECT_TRUE(jac.df_dv.isZero(0.0));
}

}  // namespace
}  // namespace grip
