#include "gradient/rollout.hpp"

#include <cassert>
#include <cstddef>

#include "dynamics/integrator.hpp"

namespace grip {

std::vector<std::vector<RigidBodyState>> rollout_system(const std::vector<RigidBodyState>& initial, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<std::vector<Eigen::Vector3d>>& controls, double dt, double gravity) {
  assert(initial.size() == params.size());
  assert(initial.size() == shapes.size());

  std::vector<std::vector<RigidBodyState>> trajectory;
  trajectory.reserve(controls.size() + 1);
  trajectory.push_back(initial);

  for (std::size_t t = 0; t < controls.size(); ++t) {
    assert(controls[t].size() == initial.size());
    trajectory.push_back(step_system(trajectory[t], params, shapes, plane, penalty, controls[t], dt, gravity));
  }
  return trajectory;
}


RolloutGradients adjoint_system(const std::vector<std::vector<RigidBodyState>>& trajectory, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<SystemStateVector>& dl_dZ, const std::vector<SystemControlVector>& dl_dU, double dt, double gravity) {
  const std::size_t horizon = dl_dU.size();
  assert(trajectory.size() == horizon + 1);
  assert(dl_dZ.size() == horizon + 1);
  assert(params.size() == shapes.size());

  const auto num_bodies = static_cast<Eigen::Index>(params.size());

  RolloutGradients gradients;
  gradients.dJ_dU.assign(horizon, SystemControlVector::Zero(3 * num_bodies));

  // Terminal condition. Z_H sits in its own stage cost and in the constraint
  // that produced it, but in no constraint's dynamics, so the recursion has
  // nowhere further to reach and the seed stands alone.
  SystemStateVector adjoint = dl_dZ[horizon];

  for (std::size_t t = horizon; t-- > 0;) {
    const SystemStepJacobians jac = step_system_jacobian(trajectory[t], params, shapes, plane, penalty, dt, gravity);

    // Both lines consume adjoint_{t+1}, so the control gradient has to be
    // read off BEFORE the adjoint is stepped back. Using adjoint_t here is
    // the classic off-by-one: dimensionally sound, silently wrong, and only
    // a finite-difference check over the whole rollout catches it.
    gradients.dJ_dU[t] = dl_dU[t] + jac.dZ_dF.transpose() * adjoint;
    adjoint = dl_dZ[t] + jac.dZ_dZ.transpose() * adjoint;
  }

  // Z_0 appears in its stage cost and inside the first step, but never as a
  // bare term the way every later state does -- there is no constraint
  // before it. That missing term is exactly why the last thing the sweep
  // computes is the gradient rather than a condition forced to zero.
  gradients.dJ_dZ0 = adjoint;
  return gradients;
}

}  // namespace grip
