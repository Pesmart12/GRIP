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


// One contact between two moving bodies.
//
// Unlike Contact, which is a vertex measured against fixed scenery, this
// one has two participants and no fixed vertex index -- the contact set
// depends on which faces are involved, so it is neither per-vertex nor
// of constant size.
//
// normal always points from the first body toward the second: the
// direction the second must move to escape. That is a normalization, not
// a fact about the geometry -- either body may own the face the normal
// came from, and reference_is_first says which. It matters because the
// two Jacobian blocks are NOT mirror images: the body owning the face
// also owns the normal, so rotating it swings the contact plane.
//
// clipped says whether the contact sits at a material vertex of the
// incident body or at a point where its edge crosses the reference
// face's extent. That distinction is invisible physically and decisive
// for the derivative: a material vertex belongs to one body, a clip
// point belongs to both, and the Jacobian differs accordingly. See
// docs/derivations/pair_detection.md.
struct PairContact {
  double signed_distance = 0.0;
  Eigen::Vector2d normal = Eigen::Vector2d::Zero();
  Eigen::Vector2d point = Eigen::Vector2d::Zero();
  bool reference_is_first = true;
  bool clipped = false;
};


// d(d)/dq for a pair contact, spanning both bodies:
// [ d/dq_first (3) | d/dq_second (3) ].
//
// One row rather than two blocks, because that is what makes Newton's
// third law automatic: the generalized force J^T lambda splits into
// equal-and-opposite halves without the signs ever being written down.
using PairJacobian = Eigen::Matrix<double, 1, 6>;


// d2(d)/dq2 over the same six degrees of freedom -- the Hessian of the
// gap, symmetric.
//
// Needed because the contact force is J^T lambda and BOTH factors depend
// on q. Differentiating it gives
//
//   df_c/dq = -k J^T J - b J^T (H v)^T + lambda H
//
// which is the half-plane formula with H in place of its single nonzero
// entry. Against a half-plane the normal was fixed and the contact point
// was material, so H had one term; here every component of J moves.
using PairHessian = Eigen::Matrix<double, 6, 6>;


// Contacts between two convex polygons, by the separating axis theorem
// followed by clipping the incident edge to the reference face's extent.
//
// Empty when the polygons are disjoint -- and unlike the half-plane
// case, that emptiness is unavoidable: there is no gap to report when
// nothing is in contact, so detection cannot stay smooth the way
// detect_contacts_body does.
//
// Otherwise one or two contacts: two when a face meets a face, one when
// a vertex meets a face. The count changes discontinuously as a body
// tips, which is one of three new non-smooth surfaces body-body contact
// introduces. See docs/derivations/pair_detection.md.
//
// Vertices must wind counterclockwise, as BodyShape's do.
std::vector<PairContact> detect_contacts_pair(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape);


// Analytic Jacobians for the contacts above, in the same order.
//
// Three shapes appear, and only the first is familiar:
//
//   n . (p - c)^perp        the incident body sweeping the vertex, exactly
//                           the half-plane form
//   n^perp . (p - r0)       the reference body swinging the face, absent
//                           against a half-plane because a fixed plane
//                           has dn/dq = 0
//   (n . e) * dt/dq         the contact point sliding along the incident
//                           edge as the clip moves, present only at
//                           clipped contacts
//
// The last vanishes when the two faces are anti-parallel, since then the
// incident edge e lies in the reference face and n . e = 0 -- which is
// why treating clip points as material vertices looks nearly right and
// is not.
//
// The translation blocks stay exactly equal and opposite through all of
// it, so Newton's third law survives clipping.
std::vector<PairJacobian> detect_contacts_pair_jacobian(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape);


// Gap Hessians for the contacts above, in the same order. Symmetric.
std::vector<PairHessian> detect_contacts_pair_hessian(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape);

}  // namespace grip
