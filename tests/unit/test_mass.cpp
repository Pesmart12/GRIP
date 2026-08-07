#include <gtest/gtest.h>

#include <Eigen/Core>

#include "core/rigid_body.hpp"
#include "dynamics/mass.hpp"

namespace grip {
namespace {

TEST(InverseMass, MatchesDiagonalOfMInverse) {
  const RigidBodyParams params{/*mass=*/2.5, /*inertia=*/0.4};

  const Eigen::Vector3d inv_mass = inverse_mass_diagonal(params);

  // M = diag(m, m, I), so M^-1 = diag(1/m, 1/m, 1/I). Both translation
  // entries are the same mass -- planar translation is isotropic.
  EXPECT_DOUBLE_EQ(inv_mass.x(), 1.0 / 2.5);
  EXPECT_DOUBLE_EQ(inv_mass.y(), 1.0 / 2.5);
  EXPECT_DOUBLE_EQ(inv_mass.z(), 1.0 / 0.4);
}

TEST(InverseMass, AsDiagonalInvertsTheMassMatrix) {
  const RigidBodyParams params{/*mass=*/3.7, /*inertia=*/1.9};

  const Eigen::Matrix3d m = Eigen::Vector3d(params.mass, params.mass, params.inertia).asDiagonal();
  const Eigen::Matrix3d m_inv = inverse_mass_diagonal(params).asDiagonal();
  const Eigen::Matrix3d product = m_inv * m;

  // The defining property, asserted against the mass matrix rather than
  // against three hand-written reciprocals: whatever M^-1 is, M^-1 M = Id.
  EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), 1e-15));
}

}  // namespace
}  // namespace grip
