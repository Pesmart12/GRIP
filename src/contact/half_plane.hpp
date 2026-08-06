#pragma once

#include <Eigen/Core>

namespace grip {

// A static half-plane obstacle in Hesse normal form: free space is
// { p : n.p >= o }, with n the UNIT normal pointing into free space --
// the direction a penetrating body must move to escape -- and o the
// plane's signed offset from the origin along that normal.
//
// The unit-length requirement is load-bearing, not decorative: signed
// distance is computed as n.p - o, which is a true distance only when
// ||n|| = 1. A scaled normal silently scales every penetration depth,
// and with it every contact force.
//
// The obstacle has no state -- it is fixed scenery, not a body, so it
// never appears in the system state vector and contributes nothing to
// any Jacobian. Ground at y = 0 is the default: normal (0, 1), offset
// 0, matching the y-up gravity convention in
// docs/derivations/symplectic_euler.md.
struct HalfPlane {
  Eigen::Vector2d normal = Eigen::Vector2d(0.0, 1.0);
  double offset = 0.0;
};

}  // namespace grip
