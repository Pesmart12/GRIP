#include "dynamics/integrator.hpp"

#include <cassert>
#include <cstddef>

#include "contact/penalty.hpp"
#include "dynamics/mass.hpp"

namespace grip {
namespace {

// gravity + contact + control, evaluated at one state. The single place the
// assembly layer decides what a "total force" is, so the force and its
// Jacobian below cannot disagree about which terms exist.
Eigen::Vector3d TotalForce(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& u, double gravity) {
  return gravity_force(params, gravity) + penalty_force_body(state, shape, plane, penalty) + u;
}


// df/dq and df/dv for the same sum. u contributes nothing: it is additive, so
// df/du = Id and that is the integrator's business, not a force law's.
ForceJacobian TotalForceJacobian(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, double gravity) {
  ForceJacobian total = gravity_force_jacobian(state.q, state.v, params, gravity);
  const ForceJacobian contact = penalty_force_body_jacobian(state, shape, plane, penalty);
  total.df_dq += contact.df_dq;
  total.df_dv += contact.df_dv;
  return total;
}

}  // namespace

RigidBodyState integrate_body(const RigidBodyState& state, const RigidBodyParams& params, const Eigen::Vector3d& force, double dt) {
  const Eigen::Vector3d inv_mass = inverse_mass_diagonal(params);

  RigidBodyState next;
  next.v = state.v + dt * force.cwiseProduct(inv_mass);
  next.q = state.q + dt * next.v;
  return next;
}


StepJacobians integrate_body_jacobian(const RigidBodyParams& params, const ForceJacobian& force_jacobian, double dt) {
  const Eigen::Matrix3d m_inv = inverse_mass_diagonal(params).asDiagonal();
  const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d dv_dq = dt * m_inv * force_jacobian.df_dq;
  const Eigen::Matrix3d dv_dv = identity3 + dt * m_inv * force_jacobian.df_dv;
  const Eigen::Matrix3d dv_df = dt * m_inv;

  const Eigen::Matrix3d dq_dq = identity3 + dt * dv_dq;
  const Eigen::Matrix3d dq_dv = dt * dv_dv;
  const Eigen::Matrix3d dq_df = dt * dv_df;

  StepJacobians jac;
  jac.dz_dz.block<3, 3>(0, 0) = dq_dq;
  jac.dz_dz.block<3, 3>(0, 3) = dq_dv;
  jac.dz_dz.block<3, 3>(3, 0) = dv_dq;
  jac.dz_dz.block<3, 3>(3, 3) = dv_dv;

  jac.dz_df.block<3, 3>(0, 0) = dq_df;
  jac.dz_df.block<3, 3>(3, 0) = dv_df;

  return jac;
}


std::vector<RigidBodyState> integrate_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<Eigen::Vector3d>& forces, double dt) {
  assert(states.size() == params.size());
  assert(states.size() == forces.size());

  std::vector<RigidBodyState> next(states.size());
  for (std::size_t i = 0; i < states.size(); ++i) {
    next[i] = integrate_body(states[i], params[i], forces[i], dt);
  }
  return next;
}


SystemStepJacobians integrate_system_jacobian(const std::vector<RigidBodyParams>& params, const SystemForceJacobian& force_jacobian, double dt) {
  const auto n = static_cast<Eigen::Index>(params.size());
  assert(force_jacobian.dF_dQ.rows() == 3 * n && force_jacobian.dF_dQ.cols() == 3 * n);
  assert(force_jacobian.dF_dV.rows() == 3 * n && force_jacobian.dF_dV.cols() == 3 * n);

  // M_sys^-1 is block-diagonal by construction -- mass never couples
  // bodies, only forces do.
  Eigen::MatrixXd inverse_mass = Eigen::MatrixXd::Zero(3 * n, 3 * n);
  for (Eigen::Index i = 0; i < n; ++i) {
    inverse_mass.block<3, 3>(3 * i, 3 * i) = inverse_mass_diagonal(params[static_cast<std::size_t>(i)]).asDiagonal();
  }

  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(3 * n, 3 * n);
  const Eigen::MatrixXd dv_dq = dt * inverse_mass * force_jacobian.dF_dQ;
  const Eigen::MatrixXd dv_dv = identity + dt * inverse_mass * force_jacobian.dF_dV;
  const Eigen::MatrixXd dq_dq = identity + dt * dv_dq;
  const Eigen::MatrixXd dq_dv = dt * dv_dv;

  SystemStepJacobians jac;
  jac.dZ_dZ = SystemStateJacobian::Zero(6 * n, 6 * n);
  jac.dZ_dF = SystemForceSensitivity::Zero(6 * n, 3 * n);

  // Z interleaves (q, v) per body while Q and V stack each separately, so
  // the four blocks have to be scattered rather than copied.
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      jac.dZ_dZ.block<3, 3>(6 * i, 6 * j) = dq_dq.block<3, 3>(3 * i, 3 * j);
      jac.dZ_dZ.block<3, 3>(6 * i, 6 * j + 3) = dq_dv.block<3, 3>(3 * i, 3 * j);
      jac.dZ_dZ.block<3, 3>(6 * i + 3, 6 * j) = dv_dq.block<3, 3>(3 * i, 3 * j);
      jac.dZ_dZ.block<3, 3>(6 * i + 3, 6 * j + 3) = dv_dv.block<3, 3>(3 * i, 3 * j);
    }
    const Eigen::Matrix3d body_inverse_mass = inverse_mass_diagonal(params[static_cast<std::size_t>(i)]).asDiagonal();
    jac.dZ_dF.block<3, 3>(6 * i, 3 * i) = dt * dt * body_inverse_mass;
    jac.dZ_dF.block<3, 3>(6 * i + 3, 3 * i) = dt * body_inverse_mass;
  }
  return jac;
}


RigidBodyState step_body(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& u, double dt, double gravity) {
  return integrate_body(state, params, TotalForce(state, params, shape, plane, penalty, u, gravity), dt);
}


StepJacobians step_body_jacobian(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, double dt, double gravity) {
  return integrate_body_jacobian(params, TotalForceJacobian(state, params, shape, plane, penalty, gravity), dt);
}


std::vector<RigidBodyState> step_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<Eigen::Vector3d>& u, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == shapes.size());
  assert(states.size() == u.size());

  // Contact is swept over the whole system before anything is stepped,
  // because a body-body contact produces force on two bodies at once and
  // cannot be attributed to either alone. Gravity and the control wrench
  // stay per-body.
  std::vector<Eigen::Vector3d> forces = penalty_forces_system(states, shapes, plane, penalty);
  for (std::size_t i = 0; i < states.size(); ++i) {
    forces[i] += gravity_force(params[i], gravity) + u[i];
  }
  return integrate_system(states, params, forces, dt);
}


SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == shapes.size());

  // Gravity is a constant wrench and u is additive, so neither
  // contributes to dF/dQ or dF/dV -- contact is the whole of it.
  static_cast<void>(gravity);
  return integrate_system_jacobian(params, penalty_forces_system_jacobian(states, shapes, plane, penalty), dt);
}

}  // namespace grip
