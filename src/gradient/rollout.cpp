#include "gradient/rollout.hpp"

#include <cassert>
#include <cstddef>
#include <utility>

#include "dynamics/integrator.hpp"

namespace grip {

std::vector<std::vector<RigidBodyState>> rollout_system(const std::vector<RigidBodyState>& initial, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<std::vector<Eigen::Vector3d>>& controls, std::size_t substeps, double dt, double gravity) {
  assert(initial.size() == params.size());
  assert(initial.size() == shapes.size());
  assert(substeps >= 1);

  std::vector<std::vector<RigidBodyState>> trajectory;
  trajectory.reserve(controls.size() + 1);
  trajectory.push_back(initial);

  for (std::size_t t = 0; t < controls.size(); ++t) {
    assert(controls[t].size() == initial.size());
    std::vector<RigidBodyState> states = step_system(trajectory[t], params, shapes, plane, penalty, controls[t], dt, gravity);
    for (std::size_t k = 1; k < substeps; ++k) {
      states = step_system(states, params, shapes, plane, penalty, controls[t], dt, gravity);
    }
    trajectory.push_back(std::move(states));
  }
  return trajectory;
}


RolloutGradients adjoint_system(const std::vector<std::vector<RigidBodyState>>& trajectory, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<std::vector<Eigen::Vector3d>>& controls, std::size_t substeps, const std::vector<SystemStateVector>& dl_dZ, const std::vector<SystemControlVector>& dl_dU, double dt, double gravity) {
  const std::size_t horizon = dl_dU.size();
  assert(trajectory.size() == horizon + 1);
  assert(controls.size() == horizon);
  assert(dl_dZ.size() == horizon + 1);
  assert(params.size() == shapes.size());
  assert(substeps >= 1);

  const auto num_bodies = static_cast<Eigen::Index>(params.size());

  RolloutGradients gradients;
  gradients.dobjective_dU.assign(horizon, SystemControlVector::Zero(3 * num_bodies));

  // Terminal condition. Z_H sits in its own stage cost and in the constraint
  // that produced it, but in no constraint's dynamics, so the recursion has
  // nowhere further to reach and the seed stands alone.
  SystemStateVector adjoint = dl_dZ[horizon];

  for (std::size_t t = horizon; t-- > 0;) {
    // Rebuild the states inside this control step. Only its start was
    // recorded, and every substep's Jacobian is taken at the state that
    // substep starts from -- getting that wrong is adjoint.md's failure
    // mode 2. The control step's end state is trajectory[t + 1] and is not
    // needed here.
    std::vector<std::vector<RigidBodyState>> inner(substeps);
    inner[0] = trajectory[t];
    for (std::size_t k = 1; k < substeps; ++k) {
      inner[k] = step_system(inner[k - 1], params, shapes, plane, penalty, controls[t], dt, gravity);
    }

    // Both lines consume the adjoint at the END of their substep, so the
    // control gradient has to be accumulated BEFORE the adjoint is stepped
    // back. Doing it after is the classic off-by-one: dimensionally sound,
    // silently wrong, and only a finite-difference check over the whole
    // rollout catches it.
    SystemControlVector control_gradient = dl_dU[t];
    for (std::size_t k = substeps; k-- > 0;) {
      const SystemStepJacobians jac = step_system_jacobian(inner[k], params, shapes, plane, penalty, dt, gravity);
      control_gradient.noalias() += jac.dZ_dF.transpose() * adjoint;
      adjoint = jac.dZ_dZ.transpose() * adjoint;
    }
    gradients.dobjective_dU[t] = control_gradient;

    // The stage-cost seed enters only at the control-step boundary.
    adjoint += dl_dZ[t];
  }

  // Z_0 appears in its stage cost and inside the first step, but never as a
  // bare term the way every later state does -- there is no constraint
  // before it. That missing term is exactly why the last thing the sweep
  // computes is the gradient rather than a condition forced to zero.
  gradients.dobjective_dZ0 = adjoint;
  return gradients;
}

}  // namespace grip
