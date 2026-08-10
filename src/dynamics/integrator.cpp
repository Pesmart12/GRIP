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


SystemStepJacobians integrate_system_jacobian(const std::vector<RigidBodyParams>& params, const std::vector<ForceJacobian>& force_jacobians, double dt) {
  assert(params.size() == force_jacobians.size());

  const auto n = static_cast<Eigen::Index>(params.size());
  SystemStepJacobians jac;
  jac.dZ_dZ = SystemStateJacobian::Zero(6 * n, 6 * n);
  jac.dZ_dF = SystemForceSensitivity::Zero(6 * n, 3 * n);

  for (Eigen::Index i = 0; i < n; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const StepJacobians body_jac = integrate_body_jacobian(params[idx], force_jacobians[idx], dt);
    jac.dZ_dZ.block<6, 6>(6 * i, 6 * i) = body_jac.dz_dz;
    jac.dZ_dF.block<6, 3>(6 * i, 3 * i) = body_jac.dz_df;
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

  std::vector<Eigen::Vector3d> forces(states.size());
  for (std::size_t i = 0; i < states.size(); ++i) {
    forces[i] = TotalForce(states[i], params[i], shapes[i], plane, penalty, u[i], gravity);
  }
  return integrate_system(states, params, forces, dt);
}


SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == shapes.size());

  std::vector<ForceJacobian> force_jacobians(states.size());
  for (std::size_t i = 0; i < states.size(); ++i) {
    force_jacobians[i] = TotalForceJacobian(states[i], params[i], shapes[i], plane, penalty, gravity);
  }
  return integrate_system_jacobian(params, force_jacobians, dt);
}

}  // namespace grip
