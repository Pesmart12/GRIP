#pragma once

#include <vector>

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "core/rigid_body.hpp"

namespace grip {

// One body vertex measured against a half-plane.
//
// phi is the signed distance (gap): > 0 separated, 0 touching, < 0
// penetrating by that depth. point is the vertex's world position --
// where a contact force would be applied, so it also fixes the moment
// arm (point - c). normal is the escape direction, constant for a
// half-plane but carried per contact because that's what a force law
// consumes.
struct Contact {
  int vertex_index = 0;
  double phi = 0.0;
  Eigen::Vector2d normal = Eigen::Vector2d::Zero();
  Eigen::Vector2d point = Eigen::Vector2d::Zero();
};

// Every vertex of the body against the plane, in vertex order --
// including separated ones (phi > 0). Reporting all of them keeps each
// phi_i a smooth linear function of q with no branching, so the
// non-smoothness enters one step later, at whatever activation test a
// force law applies. It also makes boundary margins available without a
// separate query. See docs/derivations/contact_detection.md.
std::vector<Contact> detect_contacts_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane);

// Analytic d(phi_i)/dq for each contact, in the same vertex order:
// [n_x, n_y, n . (J (p_i - c))] with J the 90-degree rotation. One row
// vector per contact, matching detect_contacts_body's output size.
std::vector<Eigen::RowVector3d> detect_contacts_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane);

}  // namespace grip
