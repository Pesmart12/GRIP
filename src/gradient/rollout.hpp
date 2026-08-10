#pragma once

#include <vector>

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// Forward rollout: H controls in, H+1 system states out, with the initial
// state first. Each entry is one body per element, indexed alongside params
// and shapes.
//
// Nothing is recorded but the states. That is enough, because
// step_system_jacobian is a pure function of the state -- the backward sweep
// recomputes each step's Jacobian rather than taping it. Costs one extra
// Jacobian evaluation per step during the sweep, saves storing a 6B x 6B
// matrix per step, and makes a gradient-free rollout strictly cheaper. See
// docs/derivations/adjoint.md.
std::vector<std::vector<RigidBodyState>> rollout_system(const std::vector<RigidBodyState>& initial, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<std::vector<Eigen::Vector3d>>& controls, double dt, double gravity = kDefaultGravity);


// Gradients of a caller-defined scalar objective with respect to the things
// a caller can actually choose: where the trajectory started, and what it was
// driven with.
struct RolloutGradients {
  SystemStateVector dJ_dZ0;                  // 6B
  std::vector<SystemControlVector> dJ_dU;    // H entries, each 3B
};


// Reverse-mode (adjoint) sweep over a trajectory produced by rollout_system.
//
// GRIP does not define the objective. The caller supplies its partial
// derivatives -- the "seeds" -- and gets back total derivatives:
//
//   dl_dZ   H+1 entries, dl_dZ[t] = d(stage cost at t)/d(Z_t)
//   dl_dU   H   entries, dl_dU[t] = d(stage cost at t)/d(U_t)
//
// A terminal-only objective is the case where every entry but dl_dZ[H] is
// zero. A cost function is a task definition and belongs to whoever is
// posing the task; what belongs here is propagating a sensitivity backwards
// through the dynamics.
//
// The recursion, derived in docs/derivations/adjoint.md:
//
//   adjoint_H = dl_dZ[H]
//   dJ_dU[t]  = dl_dU[t] + dZ_dF_t^T * adjoint_{t+1}
//   adjoint_t = dl_dZ[t] + dZ_dZ_t^T * adjoint_{t+1}
//   dJ_dZ0    = adjoint_0
//
// with each Jacobian evaluated at the state at the START of the step it
// describes. Cost is O(H) rather than forward mode's O(H^2), which is the
// entire reason this is a separate pass and not a loop over
// step_system_jacobian.
RolloutGradients adjoint_system(const std::vector<std::vector<RigidBodyState>>& trajectory, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<SystemStateVector>& dl_dZ, const std::vector<SystemControlVector>& dl_dU, double dt, double gravity = kDefaultGravity);

}  // namespace grip
