#include "dynamics/integrator.hpp"

namespace grip {

RigidBodyState symplectic_euler_step(const RigidBodyState& state,
                                      const RigidBodyParams& params,
                                      const Eigen::Vector3d& u, double dt,
                                      double gravity) {
  const Eigen::Vector3d f = gravity_force(params, gravity) + u;
  const Eigen::Vector3d inv_mass(1.0 / params.mass, 1.0 / params.mass,
                                  1.0 / params.inertia);

  RigidBodyState next;
  next.v = state.v + dt * f.cwiseProduct(inv_mass);
  next.q = state.q + dt * next.v;
  return next;
}

StepJacobians symplectic_euler_step_jacobian(const RigidBodyState& state,
                                              const RigidBodyParams& params,
                                              const Eigen::Vector3d& /*u*/,
                                              double dt, double gravity) {
  // u is accepted (it's the linearization point, mirroring
  // symplectic_euler_step's signature) but unused: every force law is
  // additive in u, so df/du = I unconditionally and doesn't depend on
  // u's actual value. See docs/derivations/integrator_jacobians.md.
  const ForceJacobian force_jac =
      gravity_force_jacobian(state.q, state.v, params, gravity);
  const Eigen::Vector3d inv_mass(1.0 / params.mass, 1.0 / params.mass,
                                  1.0 / params.inertia);
  const Eigen::Matrix3d m_inv = inv_mass.asDiagonal();
  const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d dv_dq = dt * m_inv * force_jac.df_dq;
  const Eigen::Matrix3d dv_dv = identity3 + dt * m_inv * force_jac.df_dv;
  const Eigen::Matrix3d dv_du = dt * m_inv;  // df/du = I: u enters additively

  const Eigen::Matrix3d dq_dq = identity3 + dt * dv_dq;
  const Eigen::Matrix3d dq_dv = dt * dv_dv;
  const Eigen::Matrix3d dq_du = dt * dv_du;

  StepJacobians jac;
  jac.dx_dx.block<3, 3>(0, 0) = dq_dq;
  jac.dx_dx.block<3, 3>(0, 3) = dq_dv;
  jac.dx_dx.block<3, 3>(3, 0) = dv_dq;
  jac.dx_dx.block<3, 3>(3, 3) = dv_dv;

  jac.dx_du.block<3, 3>(0, 0) = dq_du;
  jac.dx_du.block<3, 3>(3, 0) = dv_du;

  return jac;
}

}  // namespace grip
