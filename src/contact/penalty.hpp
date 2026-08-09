#pragma once

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// Contact stiffness and damping for the penalty formulation.
//
// Deliberately its own struct rather than fields on RigidBodyParams
// (mass properties, never differentiated with respect to) or on
// HalfPlane (geometry): these are properties of the contact *model*,
// and step 8 swaps models with everything else held fixed.
//
// Zero stiffness means no contact response. That's a consistent
// degenerate case rather than a sentinel -- a body with no contact
// stiffness passes through the plane. Zero damping recovers the
// undamped spring exactly, which is what every step 5a test still
// exercises.
struct PenaltyParams {
  double stiffness = 0.0;
  double damping = 0.0;
};


// Generalized contact force f_c = sum_i J_i^T lambda_i for one body
// against a half-plane, with a clamped Kelvin-Voigt law:
//
//   lambda_i = d_i < 0 ? max(0, -k*d_i - b*ddot_i) : 0
//
// a one-sided linear spring in the penetration depth plus a damper in
// the closing rate ddot_i = J_i . v. Separated vertices contribute
// nothing.
//
// The outer max is not cosmetic. Without it the damper keeps pulling
// while the contact separates, so the plane would *stick* to a
// departing body. The clamp cures that, but only on the way out -- on
// entry the damping term is already finite the instant contact begins,
// so the force jumps discontinuously. That asymmetry is real and
// characterized in tests/validation/test_contact_boundary.cpp rather
// than papered over.
//
// The result is a single wrench (fx, fy, tau) at the center of mass no
// matter how many vertices are penetrating -- J_i^T carries each scalar
// normal force to both a force and a moment about the COM, so the
// moment arm never has to be handled separately. See
// docs/derivations/penalty_contact.md.
Eigen::Vector3d penalty_force_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty);


// df_c/dq and df_c/dv for the force above.
//
// df_c/dv = -b * sum_i J_i^T J_i, symmetric and negative semidefinite:
// the damper can only remove energy, never add it.
//
// df_c/dq splits into three pieces. A material term -k * sum J_i^T J_i;
// a geometric term from the moment arm rotating under fixed force,
// touching only the (theta, theta) entry and carrying the full lambda_i
// including its damping part; and the damper's own q-dependence through
// the closing rate, which lands entirely in the third column.
//
// That third piece is an outer product of two vectors that are not
// parallel, so with b != 0 and a nonzero angular rate df_c/dq is NOT
// symmetric. At b = 0 it is exactly -Hessian(U) and symmetry is a free
// correctness check; damping is non-conservative and takes that away.
ForceJacobian penalty_force_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty);

}  // namespace grip
