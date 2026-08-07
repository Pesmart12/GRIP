#pragma once

#include <Eigen/Core>

#include "core/rigid_body.hpp"

namespace grip {

// Inverse mass matrix for a planar body, as the 3-vector of its
// diagonal entries.
//
// The body mass matrix is M = diag(m, m, I): translation resists with
// the mass along both axes, rotation resists with the moment of inertia
// about the center of mass. That it is diagonal is a consequence of
// taking inertia about the COM -- an off-COM reference point couples
// translation and rotation and fills in the off-diagonal blocks. Since
// M is diagonal, M^-1 = diag(1/m, 1/m, 1/I) and three numbers carry all
// the information a 3x3 would.
//
// This is the single place in the codebase that encodes that
// convention, so the rollout path and the gradient path cannot drift
// apart on it. Returned as a vector rather than a matrix because both
// spellings are needed and the vector converts to either:
//
//   f.cwiseProduct(inverse_mass_diagonal(params))     // vector operand
//   inverse_mass_diagonal(params).asDiagonal() * J    // matrix operand
//
// See docs/derivations/symplectic_euler.md.
inline Eigen::Vector3d inverse_mass_diagonal(const RigidBodyParams& params) {
  return Eigen::Vector3d(1.0 / params.mass, 1.0 / params.mass, 1.0 / params.inertia);
}

}  // namespace grip
