#pragma once

#include <Eigen/Core>

namespace grip {

// A static half-plane obstacle: free space is { p : normal.p >= offset }.
// normal is unit length and points INTO free space, i.e. the direction a
// penetrating body must move to escape.
//
// The obstacle has no state -- it is fixed scenery, not a body, so it
// never appears in the system state vector and contributes nothing to
// any Jacobian. Ground at y = 0 is normal = (0, 1), offset = 0, which
// matches the y-up gravity convention from
// docs/derivations/symplectic_euler.md.
struct HalfPlane {
  Eigen::Vector2d normal = Eigen::Vector2d(0.0, 1.0);
  double offset = 0.0;
};

}  // namespace grip
