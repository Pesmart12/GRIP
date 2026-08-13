#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "core/rigid_body.hpp"

namespace grip {
namespace {

// Counterclockwise, as BodyShape requires -- the outward normal of an
// edge is its direction rotated by -90 degrees, which only points
// outward for CCW winding.
BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

// Wider than it is tall, so a square resting on it is supported across
// its whole width and the clipping has something to cut.
BodyShape Platform() {
  return BodyShape{{{-1.0, -0.25}, {1.0, -0.25}, {1.0, 0.25}, {-1.0, 0.25}}};
}

RigidBodyState At(double x, double y, double theta) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(x, y, theta);
  return state;
}

TEST(PairDetection, SeparatedPolygonsReportNothing) {
  // Side by side with a clear gap. The separating axis theorem needs only
  // one axis to prove disjointness, and here several do.
  const std::vector<PairContact> contacts = detect_contacts_pair(At(0.0, 0.0, 0.0), UnitSquare(), At(3.0, 0.0, 0.0), UnitSquare());

  EXPECT_TRUE(contacts.empty());
}

TEST(PairDetection, ExactlyTouchingReportsAZeroGapRatherThanNothing) {
  // Flush at x = 0.5. The separating-axis test is strict, so zero
  // separation counts as overlapping and a contact is reported -- with
  // d = 0, which every force law treats as no force. Same convention as
  // the half-plane, where d = 0 is the separated branch.
  //
  // Reporting it rather than dropping it keeps the contact set from
  // flickering for bodies resting exactly in touch.
  const std::vector<PairContact> contacts = detect_contacts_pair(At(0.0, 0.0, 0.0), UnitSquare(), At(1.0, 0.0, 0.0), UnitSquare());

  ASSERT_FALSE(contacts.empty());
  for (const PairContact& contact : contacts) {
    EXPECT_NEAR(contact.signed_distance, 0.0, 1e-15);
  }
}

TEST(PairDetection, DegenerateShapesReportNothing) {
  // A body with no vertices is the free-flight case used throughout the
  // integrator tests, and must stay inert here too.
  EXPECT_TRUE(detect_contacts_pair(At(0.0, 0.0, 0.0), BodyShape{}, At(0.0, 0.0, 0.0), UnitSquare()).empty());
  EXPECT_TRUE(detect_contacts_pair(At(0.0, 0.0, 0.0), UnitSquare(), At(0.0, 0.0, 0.0), BodyShape{}).empty());
}

TEST(PairDetection, StackedBoxesGiveTwoContactsAtEqualDepth) {
  // A unit square resting squarely on a wider platform, overlapping by
  // 0.1. Face meets face, so one contact could not resist tipping -- the
  // same argument that made a flat box on the ground need two.
  //
  // The square is narrower than the platform, so both contacts come from
  // clipping the platform's top edge to the square's width rather than
  // from material vertices.
  const std::vector<PairContact> contacts = detect_contacts_pair(At(0.0, 0.0, 0.0), Platform(), At(0.0, 0.65, 0.0), UnitSquare());

  ASSERT_EQ(contacts.size(), 2u);
  for (const PairContact& contact : contacts) {
    EXPECT_NEAR(contact.signed_distance, -0.1, 1e-12);
    EXPECT_TRUE(contact.clipped) << "a narrower body on a wider one should clip, not land on vertices";
    // Points to the escape direction for the second body, which is up.
    EXPECT_NEAR(contact.normal.x(), 0.0, 1e-12);
    EXPECT_NEAR(contact.normal.y(), 1.0, 1e-12);
  }

  // Clipped to the square's width, so the two contacts sit at its edges.
  EXPECT_NEAR(std::abs(contacts[0].point.x() - contacts[1].point.x()), 1.0, 1e-12);
}

TEST(PairDetection, NormalPointsFromFirstTowardSecond) {
  // The normal is normalized to a direction, not left as whichever face
  // happened to win the separating-axis search. Swapping the arguments
  // must therefore reverse it exactly.
  const std::vector<PairContact> forward = detect_contacts_pair(At(0.0, 0.0, 0.0), Platform(), At(0.0, 0.65, 0.0), UnitSquare());
  const std::vector<PairContact> reversed = detect_contacts_pair(At(0.0, 0.65, 0.0), UnitSquare(), At(0.0, 0.0, 0.0), Platform());

  ASSERT_FALSE(forward.empty());
  ASSERT_EQ(forward.size(), reversed.size());
  for (std::size_t i = 0; i < forward.size(); ++i) {
    EXPECT_NEAR(forward[i].normal.x(), -reversed[i].normal.x(), 1e-12);
    EXPECT_NEAR(forward[i].normal.y(), -reversed[i].normal.y(), 1e-12);
    // The gap itself is a property of the pair, not of the argument order.
    EXPECT_NEAR(forward[i].signed_distance, reversed[i].signed_distance, 1e-12);
  }
}

TEST(PairDetection, DeeperOverlapGivesDeeperGap) {
  // Monotonicity in the obvious direction, which a sign error would
  // invert and a wrong reference face would scramble.
  double previous = 0.0;
  for (const double height : {0.70, 0.65, 0.60, 0.55}) {
    const std::vector<PairContact> contacts = detect_contacts_pair(At(0.0, 0.0, 0.0), Platform(), At(0.0, height, 0.0), UnitSquare());
    ASSERT_FALSE(contacts.empty()) << "height " << height;
    EXPECT_LT(contacts[0].signed_distance, previous) << "height " << height;
    previous = contacts[0].signed_distance;
  }
}

TEST(PairDetection, TiltedBoxReportsBothEndsOfTheClippedEdge) {
  // Rotated so a corner leads. Clipping bounds the contact along the
  // face, not across it, so the far end of a tilted incident edge can
  // sit OUTSIDE the reference plane and carry a positive gap.
  //
  // Those are kept rather than filtered, for the same reason
  // detect_contacts_body reports separated vertices: the force law gates
  // on d < 0 anyway, and dropping them would make the contact set change
  // size as a body rocks, adding a discontinuity where none is needed.
  const std::vector<PairContact> contacts = detect_contacts_pair(At(0.0, 0.0, 0.0), Platform(), At(0.1, 0.82, 0.3), UnitSquare());

  ASSERT_FALSE(contacts.empty());
  ASSERT_LE(contacts.size(), 2u);
  for (const PairContact& contact : contacts) {
    EXPECT_NEAR(contact.normal.norm(), 1.0, 1e-12);
  }
  // At least one end is genuinely penetrating, or this is not a contact.
  bool any_penetrating = false;
  for (const PairContact& contact : contacts) {
    any_penetrating = any_penetrating || contact.signed_distance < 0.0;
  }
  EXPECT_TRUE(any_penetrating);
}

TEST(PairDetection, ParallelFacesSitExactlyOnTheReferenceFlip) {
  // Worth pinning, because it is the commonest resting configuration and
  // it lands on a discontinuity.
  //
  // When two faces are parallel and overlapping, both give EXACTLY the
  // same penetration, so which body owns the reference face is a tie.
  // Tilting either way breaks it, and the winner swaps. The gap is
  // continuous across that flip -- both faces measure the same distance
  // -- but the Jacobian is not, because the rotating-normal term belongs
  // to whichever body owns the face.
  //
  // Consequence: finite-differencing a Jacobian at squarely stacked
  // boxes measures the flip, not the derivative. The validation tests
  // deliberately tilt away from it.
  const auto at_tilt = [](double tilt) {
    return detect_contacts_pair(At(0.0, 0.0, 0.0), Platform(), At(0.0, 0.65, tilt), UnitSquare());
  };
  const auto deepest = [](const std::vector<PairContact>& contacts) {
    double smallest = 0.0;
    for (const PairContact& contact : contacts) {
      smallest = std::min(smallest, contact.signed_distance);
    }
    return smallest;
  };

  // Tilting EITHER way hands the reference to the platform, because any
  // rotation of the square drives its own face deeper than the platform's
  // while leaving the platform's shallower. Exact parallel is therefore
  // not a crossing between two regimes -- it is an isolated point whose
  // tie-break disagrees with its entire neighbourhood.
  ASSERT_FALSE(at_tilt(0.0).empty());
  EXPECT_FALSE(at_tilt(0.0).front().reference_is_first) << "the tie breaks toward the second body";
  EXPECT_TRUE(at_tilt(1.0e-3).front().reference_is_first);
  EXPECT_TRUE(at_tilt(-1.0e-3).front().reference_is_first);

  // So the analytic Jacobian at exact parallel is not the limit of the
  // Jacobian from either side, and finite-differencing there compares two
  // different formulas. The gap does not jump with it: penetration
  // deepens symmetrically with tilt, whichever way.
  EXPECT_NEAR(deepest(at_tilt(1.0e-6)), deepest(at_tilt(-1.0e-6)), 1e-12);
  EXPECT_NEAR(deepest(at_tilt(0.0)), -0.1, 1e-12);
}

}  // namespace
}  // namespace grip
