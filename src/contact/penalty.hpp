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
// undamped spring exactly, and zero slip_damping recovers the
// frictionless case, so each addition is a no-op at its default.
//
// slip_damping (b_slip in the derivations) resists sliding along the
// surface, in the same units as damping. friction is Coulomb's mu, and
// bounds that resistance by mu * lambda -- the friction cone. Its
// half-angle atan(mu) is the steepest slope a body rests on, so
// mu = 0.5 holds up to 26.6 degrees.
//
// mu = 0 is frictionless regardless of slip_damping: the cone collapses
// to a line and no tangential force can be produced. That is the
// physically right meaning, and it is why every step 1-7 test still
// passes untouched.
//
// See docs/derivations/penalty_contact.md.
struct PenaltyParams {
  double stiffness = 0.0;
  double damping = 0.0;
  double slip_damping = 0.0;
  double friction = 0.0;
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
// Alongside it, a tangential force resisting slip:
//
//   beta_i = -b_slip * s_i,        s_i = J_perp,i . v
//
// applied only where the normal force is nonzero -- no contact, no
// friction -- so it switches on and off with lambda rather than on its
// own. The full contact force is then
//
//   f_c = sum_i [ J_i^T lambda_i + J_perp,i^T beta_i ]
//
// The result is a single wrench (fx, fy, tau) at the center of mass no
// matter how many vertices are penetrating -- J_i^T carries each scalar
// force to both a force and a moment about the COM, so the moment arm
// never has to be handled separately. See
// docs/derivations/penalty_contact.md.
Eigen::Vector3d penalty_force_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty);


// df_c/dq and df_c/dv for the force above.
//
// df_c/dv = -b * sum_i J_i^T J_i - b_slip * sum_i J_perp,i^T J_perp,i,
// symmetric and negative semidefinite: neither damper can add energy.
// Stacking both directions makes this -A^T diag(b, b_slip) A with A the
// normal and perp rows together, so the determinant identity picks up a
// Delassus operator over both directions rather than just the normal
// one. The Coulomb cone will break this symmetry; it holds while the
// tangential force is unbounded.
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


// Contact forces for a whole system: every body against the plane, plus
// every pair of bodies against each other. One generalized force per
// body, indexed alongside states and shapes.
//
// This exists because body-body contact does not decompose. A contact
// between bodies i and j puts +J^T lambda on one and -J^T lambda on the
// other, so there is no such thing as "body i's contact force" computed
// from body i alone -- the whole system has to be swept before any body
// can be stepped.
//
// Pair contacts carry the same clamped Kelvin-Voigt normal force and the
// same Coulomb cone as contacts against the plane. Each manifold point
// carries its own cone, bounded by its own lambda, so a resting box can
// stick at one corner while sliding at the other.
std::vector<Eigen::Vector3d> penalty_forces_system(const std::vector<RigidBodyState>& states, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty);


// dF/dQ and dF/dV for the above, over the stacked system.
//
// Plane contacts land on the diagonal blocks, exactly as before. Pair
// contacts write to all four blocks of their two bodies, and the
// off-diagonal ones are the first genuine coupling in the project --
// what multi_body_system.md has been predicting since step 3.
//
// Each pair contact contributes, over its six shared degrees of freedom,
//
//   df/dq = -k J^T J - b J^T (H v)^T + lambda H
//   df/dv = -b J^T J
//
// with H the gap Hessian. That is the half-plane formula with H in place
// of its single nonzero entry.
SystemForceJacobian penalty_forces_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty);

}  // namespace grip
