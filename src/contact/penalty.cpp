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

}  // namespace

Eigen::Vector3d penalty_force_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, plane);

  Eigen::Vector3d force = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < contacts.size(); ++i) {
  const double closing_rate = jacobians[i].dot(state.v);
    const double normal_force = NormalForceMagnitude(contacts[i].signed_distance, closing_rate, penalty);
    force += jacobians[i].transpose() * normal_force;
  }
  return force;
}


ForceJacobian penalty_force_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, plane);
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
  }
  return jac;
}

}  // namespace grip
