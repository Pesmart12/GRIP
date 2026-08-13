#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "core/rigid_body.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

BodyShape Platform() {
  return BodyShape{{{-1.0, -0.25}, {1.0, -0.25}, {1.0, 0.25}, {-1.0, 0.25}}};
}

RigidBodyState At(double x, double y, double theta) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(x, y, theta);
  return state;
}

// Central-difference d(d)/d(q_first, q_second) for one contact, by
// re-running real detection at perturbed configurations of both bodies.
PairJacobian FiniteDifferencePairJacobian(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape, std::size_t contact_index) {
  const std::function<Eigen::Matrix<double, 1, 1>(const Eigen::Matrix<double, 6, 1>&)> distance_of_q = [&](const Eigen::Matrix<double, 6, 1>& q) {
    RigidBodyState perturbed_first = first;
    RigidBodyState perturbed_second = second;
    perturbed_first.q = q.head<3>();
    perturbed_second.q = q.tail<3>();
    Eigen::Matrix<double, 1, 1> out;
    out(0) = detect_contacts_pair(perturbed_first, first_shape, perturbed_second, second_shape)[contact_index].signed_distance;
    return out;
  };
  Eigen::Matrix<double, 6, 1> q;
  q << first.q, second.q;
  return testutil::CentralDifferenceJacobian<1, 6>(distance_of_q, q);
}

void ExpectJacobianMatchesFiniteDifference(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape) {
  const std::vector<PairContact> contacts = detect_contacts_pair(first, first_shape, second, second_shape);
  const std::vector<PairJacobian> analytic = detect_contacts_pair_jacobian(first, first_shape, second, second_shape);

  ASSERT_FALSE(contacts.empty());
  ASSERT_EQ(analytic.size(), contacts.size());

  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const PairJacobian fd = FiniteDifferencePairJacobian(first, first_shape, second, second_shape, i);
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(analytic[i](j), fd(j), 1e-6) << "contact " << i << (contacts[i].clipped ? " (clipped)" : " (vertex)") << ", component " << j;
    }
  }
}

// Perfectly parallel faces give both bodies exactly the same penetration,
// so which one owns the reference face is a tie that any perturbation
// breaks -- see PairDetection.ParallelFacesSitExactlyOnTheReferenceFlip.
// Finite differences there measure the flip rather than the derivative,
// so every configuration below is tilted clear of it.
TEST(PairJacobians, MatchCentralFiniteDifferenceJustOffParallel) {
  // A degree or so from square. Still a face-on-face contact with two
  // clipped points, but off the tie, so the derivative exists.
  ExpectJacobianMatchesFiniteDifference(At(0.0, 0.0, 0.02), Platform(), At(0.0, 0.65, 0.0), UnitSquare());
}

TEST(PairJacobians, MatchCentralFiniteDifferenceWhenTheFacesAreNotParallel) {
  // Tilted, so n . e != 0 and the clip corrections are genuinely active.
  // This is the configuration that distinguishes a correct clip Jacobian
  // from one that treats the contact point as a material vertex.
  ExpectJacobianMatchesFiniteDifference(At(0.0, 0.0, 0.0), Platform(), At(0.1, 0.82, 0.3), UnitSquare());
}

TEST(PairJacobians, MatchCentralFiniteDifferenceWithBothBodiesRotated) {
  // Neither body axis-aligned, so every block has all three components
  // nonzero and a dropped rotation term cannot hide behind a zero.
  ExpectJacobianMatchesFiniteDifference(At(-0.2, 0.1, -0.25), Platform(), At(0.25, 0.78, 0.45), UnitSquare());
}

TEST(PairJacobians, MatchCentralFiniteDifferenceWithArgumentsSwapped) {
  // The reference face belongs to whichever body wins the separating-axis
  // search, not to whichever was passed first, so the two blocks are not
  // interchangeable. Swapping must still agree with finite differences.
  ExpectJacobianMatchesFiniteDifference(At(0.1, 0.82, 0.3), UnitSquare(), At(0.0, 0.0, 0.0), Platform());
}

TEST(PairJacobians, TranslationBlocksAreExactlyEqualAndOpposite) {
  // Newton's third law, before any force law is involved. The generalized
  // force J^T lambda splits into equal-and-opposite halves only because
  // the two translation blocks are exact negations -- and that survives
  // clipping, where the contact point depends on both bodies at once.
  const std::vector<std::pair<RigidBodyState, RigidBodyState>> configurations = {
      {At(0.0, 0.0, 0.0), At(0.0, 0.65, 0.0)},
      {At(0.0, 0.0, 0.0), At(0.1, 0.82, 0.3)},
      {At(-0.2, 0.1, -0.25), At(0.25, 0.78, 0.45)},
  };

  for (const auto& [first, second] : configurations) {
    const std::vector<PairJacobian> analytic = detect_contacts_pair_jacobian(first, Platform(), second, UnitSquare());
    ASSERT_FALSE(analytic.empty()) << first.q.transpose() << " / " << second.q.transpose();

    for (const PairJacobian& row : analytic) {
      EXPECT_DOUBLE_EQ(row(0), -row(3));
      EXPECT_DOUBLE_EQ(row(1), -row(4));
    }
  }
}

// Central-difference Hessian: finite-difference the analytic Jacobian,
// which is the only independent handle on a second derivative.
PairHessian FiniteDifferencePairHessian(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape, std::size_t contact_index) {
  const std::function<Eigen::Matrix<double, 6, 1>(const Eigen::Matrix<double, 6, 1>&)> jacobian_of_q = [&](const Eigen::Matrix<double, 6, 1>& q) {
    RigidBodyState perturbed_first = first;
    RigidBodyState perturbed_second = second;
    perturbed_first.q = q.head<3>();
    perturbed_second.q = q.tail<3>();
    return Eigen::Matrix<double, 6, 1>(detect_contacts_pair_jacobian(perturbed_first, first_shape, perturbed_second, second_shape)[contact_index].transpose());
  };
  Eigen::Matrix<double, 6, 1> q;
  q << first.q, second.q;
  return testutil::CentralDifferenceJacobian<6, 6>(jacobian_of_q, q);
}

void ExpectHessianMatchesFiniteDifference(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape, double tolerance) {
  const std::vector<PairContact> contacts = detect_contacts_pair(first, first_shape, second, second_shape);
  const std::vector<PairHessian> analytic = detect_contacts_pair_hessian(first, first_shape, second, second_shape);

  ASSERT_FALSE(contacts.empty());
  ASSERT_EQ(analytic.size(), contacts.size());

  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const PairHessian fd = FiniteDifferencePairHessian(first, first_shape, second, second_shape, i);
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        EXPECT_NEAR(analytic[i](a, b), fd(a, b), tolerance) << "contact " << i << (contacts[i].clipped ? " (clipped)" : " (vertex)") << " at (" << a << ", " << b << ")";
      }
    }
  }
}

TEST(PairHessians, MatchCentralFiniteDifferenceOfTheJacobian) {
  // Differencing an analytic first derivative loses roughly half the
  // available precision, so the tolerance is looser than the Jacobian
  // tests' -- but it is still six independent digits on every one of
  // thirty-six entries.
  ExpectHessianMatchesFiniteDifference(At(0.0, 0.0, 0.02), Platform(), At(0.0, 0.65, 0.0), UnitSquare(), 1e-5);
  ExpectHessianMatchesFiniteDifference(At(0.0, 0.0, 0.0), Platform(), At(0.1, 0.82, 0.3), UnitSquare(), 1e-5);
  ExpectHessianMatchesFiniteDifference(At(-0.2, 0.1, -0.25), Platform(), At(0.25, 0.78, 0.45), UnitSquare(), 1e-5);
}

TEST(PairHessians, AreSymmetric) {
  // A Hessian of a scalar must be symmetric. It is not assembled that
  // way -- every entry comes out of the composition independently -- so
  // this is a real check on the product rules rather than a tautology.
  const std::vector<PairHessian> hessians = detect_contacts_pair_hessian(At(-0.2, 0.1, -0.25), Platform(), At(0.25, 0.78, 0.45), UnitSquare());

  ASSERT_FALSE(hessians.empty());
  for (const PairHessian& hessian : hessians) {
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < a; ++b) {
        EXPECT_NEAR(hessian(a, b), hessian(b, a), 1e-12) << "at (" << a << ", " << b << ")";
      }
    }
  }
}

TEST(PairHessians, TranslationOnlyBlocksVanish) {
  // The gap is affine in either body's position, so all four
  // translation-translation entries are identically zero. The half-plane
  // case has the same property, asserted there as the translation block
  // being exactly the normal.
  const std::vector<PairHessian> hessians = detect_contacts_pair_hessian(At(-0.2, 0.1, -0.25), Platform(), At(0.25, 0.78, 0.45), UnitSquare());

  ASSERT_FALSE(hessians.empty());
  for (const PairHessian& hessian : hessians) {
    for (const int a : {0, 1, 3, 4}) {
      for (const int b : {0, 1, 3, 4}) {
        EXPECT_DOUBLE_EQ(hessian(a, b), 0.0) << "at (" << a << ", " << b << ")";
      }
    }
  }
}

TEST(PairJacobians, TranslationBlockIsTheContactNormal) {
  // The gap is linear in either body's position, so its translation
  // derivatives are exactly the normal -- not approximately, exactly.
  // The half-plane case asserts the same thing for its one body.
  const RigidBodyState first = At(0.0, 0.0, 0.0);
  const RigidBodyState second = At(0.1, 0.82, 0.3);

  const std::vector<PairContact> contacts = detect_contacts_pair(first, Platform(), second, UnitSquare());
  const std::vector<PairJacobian> analytic = detect_contacts_pair_jacobian(first, Platform(), second, UnitSquare());

  ASSERT_FALSE(contacts.empty());
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    EXPECT_NEAR(analytic[i](3), contacts[i].normal.x(), 1e-12) << "contact " << i;
    EXPECT_NEAR(analytic[i](4), contacts[i].normal.y(), 1e-12) << "contact " << i;
  }
}

}  // namespace
}  // namespace grip
