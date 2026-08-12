#include "contact/penalty.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "contact/detection.hpp"

namespace grip {
namespace {

// lambda_i = d_i < 0 ? max(0, -k*d_i - b*ddot_i) : 0.
//
// Two separate gates, and they do different jobs. The d_i < 0 test is
// activation: a separated vertex feels nothing however fast it moves.
// The max is the adhesion clamp: while a penetrating vertex is pulling
// away fast enough, the damper alone would make lambda negative, which
// would have the plane holding the body down.
//
// Both boundaries return zero exactly, and the Jacobian below treats
// that same zero as contributing nothing. The derivative genuinely does
// not exist at either one, so the convention is to report the branch
// that matches the force value. See
// docs/derivations/penalty_contact.md.
double NormalForceMagnitude(double signed_distance, double closing_rate, const PenaltyParams& penalty) {
  if (signed_distance >= 0.0) {
    return 0.0;
  }
  return std::max(0.0, -penalty.stiffness * signed_distance - penalty.damping * closing_rate);
}


// beta_i = -clamp(b_slip * s_i, -mu*lambda_i, +mu*lambda_i), and zero
// where there is no normal force to speak of. Gating on lambda rather
// than on d_i means friction switches off with the contact rather than a
// step later, and mu*lambda would force it to zero there anyway.
//
// Written as a saturated linear function of slip rather than as
// -mu*lambda*sign(s). The two agree while sliding, but sign() jumps by
// the full 2*mu*lambda across s = 0 -- precisely the state a resting
// body sits in. The clamp form is continuous there, with slope -b_slip
// on both sides, so it introduces no boundary at zero slip at all. The
// only new non-smooth surface is the cone itself, and that one is a kink
// rather than a jump: at saturation both branches give the same force.
//
// The cone bound holds exactly by construction. What is approximated is
// sticking: instead of s = 0, a held contact creeps at beta/b_slip. See
// docs/derivations/penalty_contact.md.
double TangentialForceMagnitude(double normal_force, double slip_rate, const PenaltyParams& penalty) {
  if (normal_force <= 0.0) {
    return 0.0;
  }
  const double bound = penalty.friction * normal_force;
  return -std::clamp(penalty.slip_damping * slip_rate, -bound, bound);
}

}  // namespace

Eigen::Vector3d penalty_force_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, plane);
  const std::vector<Eigen::RowVector3d> perp_jacobians = detect_contacts_body_perp_jacobian(state, shape, plane);

  Eigen::Vector3d force = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const double closing_rate = jacobians[i].dot(state.v);
    const double normal_force = NormalForceMagnitude(contacts[i].signed_distance, closing_rate, penalty);
    force += jacobians[i].transpose() * normal_force;

    const double slip_rate = perp_jacobians[i].dot(state.v);
    const double tangential_force = TangentialForceMagnitude(normal_force, slip_rate, penalty);
    force += perp_jacobians[i].transpose() * tangential_force;
  }
  return force;
}


ForceJacobian penalty_force_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, plane);
  const std::vector<Eigen::RowVector3d> perp_jacobians = detect_contacts_body_perp_jacobian(state, shape, plane);
  const Eigen::Vector2d tangent = Eigen::Vector2d(-plane.normal.y(), plane.normal.x());
  const double angular_rate = state.v.z();

  ForceJacobian jac;  // both blocks default to zero
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const double closing_rate = jacobians[i].dot(state.v);
    const double normal_force = NormalForceMagnitude(contacts[i].signed_distance, closing_rate, penalty);
    if (normal_force <= 0.0) {
      continue;  // separated, or held at zero by the adhesion clamp
    }

    const Eigen::Vector2d arm = contacts[i].point - state.q.head<2>();
    const double normal_dot_arm = plane.normal.dot(arm);

    // Material stiffness, from d(lambda_i)/dq = -k * J_i.
    jac.df_dq.noalias() -= penalty.stiffness * jacobians[i].transpose() * jacobians[i];

    // The damper's own q-dependence, through the closing rate. d(J_i.v)/dq
    // has a single nonzero entry, at theta, equal to -omega * n . (p_i - c),
    // so this whole contribution lands in the third column. It is an outer
    // product of J_i^T with a vector that is not parallel to J_i, which is
    // exactly what makes df_c/dq asymmetric once b != 0.
    jac.df_dq.col(2) += penalty.damping * angular_rate * normal_dot_arm * jacobians[i].transpose();

    // Geometric stiffness, from the moment arm rotating while the force
    // stays fixed -- perp applied twice is negation. This carries the
    // full lambda_i, damping included, because it comes from
    // differentiating J_i^T in the product J_i^T lambda_i.
    jac.df_dq(2, 2) -= normal_force * normal_dot_arm;

    // Damping, from d(lambda_i)/dv = -b * J_i. Symmetric and negative
    // semidefinite by construction, so v . f_damp <= 0 always.
    jac.df_dv.noalias() -= penalty.damping * jacobians[i].transpose() * jacobians[i];

    const double slip_rate = perp_jacobians[i].dot(state.v);
    const double tangential_force = TangentialForceMagnitude(normal_force, slip_rate, penalty);
    const double tangent_dot_arm = tangent.dot(arm);
    const double unclamped = penalty.slip_damping * slip_rate;
    const double bound = penalty.friction * normal_force;

    if (std::abs(unclamped) > bound) {
      // Sliding, pinned to the cone: beta_i = -sigma * mu * lambda_i.
      // Its whole dependence on state now runs through the NORMAL force,
      // so the derivatives are mu times lambda's, carried across by
      // J_perp^T. That produces an outer product of two vectors that are
      // not parallel and not even in the same direction -- which is what
      // makes df_c/dv asymmetric, and no longer of the form -A^T D A
      // that the generalized Delassus determinant relies on.
      const double sign = (unclamped > 0.0) ? 1.0 : -1.0;
      const double scale = sign * penalty.friction;

      jac.df_dq.noalias() += scale * penalty.stiffness * perp_jacobians[i].transpose() * jacobians[i];
      jac.df_dq.col(2) -= scale * penalty.damping * angular_rate * normal_dot_arm * perp_jacobians[i].transpose();
      jac.df_dv.noalias() += scale * penalty.damping * perp_jacobians[i].transpose() * jacobians[i];
    } else {
      // Sticking, below the bound: the tangential term repeats all the
      // normal shapes one direction over, n^perp for n throughout, with
      // the same perp-twice-is-negation argument behind
      // d(J_perp,i)/dtheta.
      jac.df_dq.col(2) += penalty.slip_damping * angular_rate * tangent_dot_arm * perp_jacobians[i].transpose();
      jac.df_dv.noalias() -= penalty.slip_damping * perp_jacobians[i].transpose() * perp_jacobians[i];
    }

    // Geometric stiffness from J_perp,i rotating under a fixed force.
    // Independent of which branch produced beta_i, so it sits outside.
    jac.df_dq(2, 2) -= tangential_force * tangent_dot_arm;
  }
  return jac;
}

}  // namespace grip
