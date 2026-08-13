#pragma once

#include <vector>

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// ===========================================================================
// Integration
//
// These four functions are the whole integrator, and they know nothing about
// contact -- not polygons, not half-planes, not stiffness. They take a force
// that somebody else assembled and advance the state.
//
// That separation is what lets contact change without the numerical core
// changing with it. It also becomes load-bearing at the body-body step:
// a contact between two bodies puts +J^T lambda on one and -J^T lambda on the
// other, so contact forces stop decomposing per body and have to be assembled
// over the system before any integration happens.
// ===========================================================================

// One symplectic (semi-implicit) Euler step for a single body, given the
// total generalized force already summed:
//
//   v_{t+1} = v_t + dt * M^-1 * force
//   q_{t+1} = q_t + dt * v_{t+1}
//
// The caller evaluates the force at the OLD state; the position update then
// uses the NEW velocity. That ordering, not the force law, is what makes the
// scheme symplectic. See docs/derivations/symplectic_euler.md.
RigidBodyState integrate_body(const RigidBodyState& state, const RigidBodyParams& params, const Eigen::Vector3d& force, double dt);


// z = (q, v) stacked per StateVector's ordering (core/rigid_body.hpp).
using StateJacobian = Eigen::Matrix<double, 6, 6>;


// d(z_{t+1})/d(force): how the step responds to the force it was handed.
//
// This is also exactly the control Jacobian d(z_{t+1})/du, because u enters
// every force law additively and so df/du = Id unconditionally. They are the
// same matrix, not merely equal in value -- which is why the integrator does
// not need to see u at all. See docs/derivations/integrator_jacobians.md.
using ForceSensitivity = Eigen::Matrix<double, 6, 3>;


struct StepJacobians {
  StateJacobian dz_dz;
  ForceSensitivity dz_df;
};


// Analytic Jacobians of integrate_body.
//
// Deliberately takes no state. The integrator's Jacobian depends only on the
// mass matrix, the timestep, and the force law's derivatives -- never on the
// operating point directly. That was true before this signature said so; now
// the type carries it.
StepJacobians integrate_body_jacobian(const RigidBodyParams& params, const ForceJacobian& force_jacobian, double dt);


// N bodies, each integrated with integrate_body above. forces is indexed
// alongside states and params. No coupling happens here -- if two bodies
// interact, that already happened when their forces were assembled.
std::vector<RigidBodyState> integrate_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<Eigen::Vector3d>& forces, double dt);


// System state Z = concatenation of each body's (q, v), per
// SystemStateVector's convention (core/rigid_body.hpp): body i at
// [6i, 6i+6).
using SystemStateJacobian = Eigen::MatrixXd;
using SystemForceSensitivity = Eigen::MatrixXd;


struct SystemStepJacobians {
  SystemStateJacobian dZ_dZ;
  SystemForceSensitivity dZ_dF;
};


// The system chain rule, which is the single-body one with 3B-dimensional
// blocks:
//
//   dV_dQ = dt * M_sys^-1 * dF/dQ        dV_dV = Id + dt * M_sys^-1 * dF/dV
//   dQ_dQ = Id + dt * dV_dQ              dQ_dV = dt * dV_dV
//
// Nothing here knows whether the forces couple. dZ_dZ comes out
// block-diagonal when dF/dQ is and dense when it is not, so body-body
// contact changes the contents and not the assembly -- which is what
// multi_body_system.md predicted when this was still a per-body loop.
//
// dZ_dF stays block-diagonal regardless: body i's force moves body i's state
// and nothing else, whatever produced that force.
SystemStepJacobians integrate_system_jacobian(const std::vector<RigidBodyParams>& params, const SystemForceJacobian& force_jacobian, double dt);


// ===========================================================================
// Assembly
//
// Thin convenience over the above: sum gravity, penalty contact and the
// control wrench, then integrate. Provided so the common path is a single
// call that cannot evaluate the force and its Jacobian at different states.
//
// Deliberately stateless and deliberately not a stability promise. The public
// API arrives with the bindings work and will likely replace these with a
// scene-level entry point; nothing here should accumulate state or caching in
// the meantime, or it stops being cheap to discard.
// ===========================================================================

// A body with no vertices has no contacts, so BodyShape{} makes the contact
// term identically zero and recovers free flight.
RigidBodyState step_body(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& u, double dt, double gravity = kDefaultGravity);


StepJacobians step_body_jacobian(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, double dt, double gravity = kDefaultGravity);


// Multiple bodies against the same static half-plane. They do not couple to
// each other -- the plane is scenery, not a body, so each body's contact
// force reads only its own state. shapes is indexed alongside params; plane
// and penalty are shared.
std::vector<RigidBodyState> step_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<Eigen::Vector3d>& u, double dt, double gravity = kDefaultGravity);


SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, double dt, double gravity = kDefaultGravity);

}  // namespace grip
