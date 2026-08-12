#pragma once

#include <vector>

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "core/rigid_body.hpp"

namespace grip {

// One body vertex measured against a half-plane.
//
// signed_distance (d in the derivations) is the gap: > 0 separated,
// 0 touching, < 0 penetrating by that depth. point is the vertex's
// world position -- where a contact force would be applied, so it also
// fixes the moment arm (point - c). normal is the escape direction,
// constant for a half-plane but carried per contact because that's what
// a force law consumes.
struct Contact {
  int vertex_index = 0;
  double signed_distance = 0.0;
  Eigen::Vector2d normal = Eigen::Vector2d::Zero();
  Eigen::Vector2d point = Eigen::Vector2d::Zero();
};


// Every vertex of the body against the plane, in vertex order --
// including separated ones. Reporting all of them keeps each d_i a
// smooth linear function of q with no branching, so the non-smoothness
// enters one step later, at whatever activation test a force law
// applies. It also makes boundary margins available without a separate
// query. See docs/derivations/contact_detection.md.
std::vector<Contact> detect_contacts_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane);


// Analytic contact Jacobian J_i = d(d_i)/dq for each contact, in the
// same vertex order: [n_x, n_y, n . (p_i - c)^perp]. One row vector per
// contact, matching detect_contacts_body's output size.
std::vector<Eigen::RowVector3d> detect_contacts_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane);


// The same row, one direction over: the contact's sensitivity along
// n^perp rather than along n. Same vertex order and same size.
//
//   J_perp,i = [(n^perp)_x, (n^perp)_y, n^perp . (p_i - c)^perp]
//
// Unlike J_i this is NOT the gradient of anything. There is no
// tangential gap to differentiate -- without a stick anchor, sliding has
// no accumulated position. What it is instead is the map from
// generalized velocity to slip speed,
//
//   s_i = J_perp,i . v
//
// which expands to n^perp . (v_xy + omega * (p_i - c)^perp): the world
// velocity of the contact point, projected onto the surface. Zero means
// sticking; nonzero means sliding, and friction opposes it.
//
// Sign convention: n^perp = (-n_y, n_x), so for the default ground plane
// n = (0, 1) it points in -x, and a body sliding in +x has NEGATIVE slip.
// Self-consistent, but it reads backwards -- see
// docs/derivations/penalty_contact.md.
std::vector<Eigen::RowVector3d> detect_contacts_body_perp_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane);

}  // namespace grip
