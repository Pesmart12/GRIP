#include "contact/detection.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

#include <Eigen/Geometry>

namespace grip {
namespace {

// Perp operator: a^perp = (-a_y, a_x), rotation by +90 degrees.
// dR/dtheta * r = (R(theta) r)^perp, so a vertex's velocity under unit
// angular rate is (p - c)^perp. See docs/derivations/notation.md.
Eigen::Vector2d Perp(const Eigen::Vector2d& a) {
  return Eigen::Vector2d(-a.y(), a.x());
}


std::vector<Eigen::Vector2d> WorldVertices(const RigidBodyState& state, const BodyShape& shape) {
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();
  std::vector<Eigen::Vector2d> world(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    world[i] = state.q.head<2>() + rotation * shape.vertices[i];
  }
  return world;
}


// Outward normal of the edge leaving vertex i, for counterclockwise
// winding. Perp() rotates by +90 degrees, which points inward for CCW,
// so the outward normal is its negation.
Eigen::Vector2d OutwardNormal(const std::vector<Eigen::Vector2d>& polygon, std::size_t i) {
  const Eigen::Vector2d edge = polygon[(i + 1) % polygon.size()] - polygon[i];
  return Eigen::Vector2d(edge.y(), -edge.x()).normalized();
}


struct FaceQuery {
  double separation = 0.0;
  std::size_t face = 0;
};


// The separating axis theorem, one direction. For each face of `owner`,
// how far the nearest vertex of `other` sits outside it; the answer is
// the LARGEST such value, over faces.
//
// Positive means some face separates the two, so they are disjoint --
// that is the theorem, and for convex polygons the face normals of both
// bodies are a sufficient set of candidate axes. Negative means every
// face is penetrated, and the least-penetrated one is the contact
// normal: the shortest way out.
//
// Ties are broken by taking the first face, with a strict comparison, so
// the result is deterministic rather than dependent on iteration
// details.
FaceQuery DeepestFace(const std::vector<Eigen::Vector2d>& owner, const std::vector<Eigen::Vector2d>& other) {
  FaceQuery best;
  best.separation = -std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < owner.size(); ++i) {
    const Eigen::Vector2d normal = OutwardNormal(owner, i);
    double nearest = std::numeric_limits<double>::max();
    for (const Eigen::Vector2d& vertex : other) {
      nearest = std::min(nearest, normal.dot(vertex - owner[i]));
    }
    if (nearest > best.separation) {
      best.separation = nearest;
      best.face = i;
    }
  }
  return best;
}


// The face of `polygon` most squarely facing `normal` -- the one whose
// outward normal is closest to anti-parallel. That is the face that
// actually meets the reference face, so it is the one to clip.
std::size_t IncidentFace(const std::vector<Eigen::Vector2d>& polygon, const Eigen::Vector2d& normal) {
  std::size_t best = 0;
  double most_opposed = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const double alignment = OutwardNormal(polygon, i).dot(normal);
    if (alignment < most_opposed) {
      most_opposed = alignment;
      best = i;
    }
  }
  return best;
}


// One surviving end of the clipped incident edge, in the edge's own
// parameter: point = incident_start + parameter * (incident_end -
// incident_start).
//
// clipped distinguishes the two cases the derivative cares about. At
// parameter 0 or 1 the contact is a material vertex of the incident
// body and the parameter is a constant. Anywhere between, the parameter
// was set by a reference-face side plane, so it depends on BOTH bodies'
// configurations and has to be differentiated too.
struct ClippedPoint {
  double parameter = 0.0;
  bool clipped = false;
  Eigen::Vector2d clip_origin = Eigen::Vector2d::Zero();
};


// Everything about a pair contact that either public entry point needs,
// computed once. detect_contacts_pair projects it down to the physics;
// the Jacobian uses all of it.
struct PairGeometry {
  bool reference_is_first = true;
  Eigen::Vector2d normal = Eigen::Vector2d::Zero();
  Eigen::Vector2d face_start = Eigen::Vector2d::Zero();
  Eigen::Vector2d face_end = Eigen::Vector2d::Zero();
  Eigen::Vector2d incident_start = Eigen::Vector2d::Zero();
  Eigen::Vector2d incident_end = Eigen::Vector2d::Zero();
  std::vector<ClippedPoint> points;
};


PairGeometry ComputePairGeometry(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape) {
  PairGeometry geometry;
  if (first_shape.vertices.size() < 3 || second_shape.vertices.size() < 3) {
    return geometry;  // a polygon needs at least a triangle
  }

  const std::vector<Eigen::Vector2d> first_world = WorldVertices(first, first_shape);
  const std::vector<Eigen::Vector2d> second_world = WorldVertices(second, second_shape);

  const FaceQuery from_first = DeepestFace(first_world, second_world);
  const FaceQuery from_second = DeepestFace(second_world, first_world);

  // A single separating axis proves disjointness, and for convex
  // polygons the two bodies' face normals are a sufficient set to search.
  if (from_first.separation > 0.0 || from_second.separation > 0.0) {
    return geometry;
  }

  // The least-penetrated face is the shortest way out, so it carries the
  // contact normal. Which body owns it is not fixed, and it flips
  // discontinuously as the configuration changes.
  geometry.reference_is_first = from_first.separation > from_second.separation;
  const std::vector<Eigen::Vector2d>& reference = geometry.reference_is_first ? first_world : second_world;
  const std::vector<Eigen::Vector2d>& incident = geometry.reference_is_first ? second_world : first_world;
  const std::size_t face = geometry.reference_is_first ? from_first.face : from_second.face;

  geometry.normal = OutwardNormal(reference, face);
  geometry.face_start = reference[face];
  geometry.face_end = reference[(face + 1) % reference.size()];

  const std::size_t incident_face = IncidentFace(incident, geometry.normal);
  geometry.incident_start = incident[incident_face];
  geometry.incident_end = incident[(incident_face + 1) % incident.size()];

  // Clip the incident edge to the reference face's extent. Both side
  // planes are linear in the edge parameter, so this is an interval
  // intersection rather than a geometric construction:
  //
  //   tangent . (i0 + t*e - r0) >= 0        and        ... - r1 <= 0
  const Eigen::Vector2d tangent = (geometry.face_end - geometry.face_start).normalized();
  const Eigen::Vector2d edge = geometry.incident_end - geometry.incident_start;
  const double slope = tangent.dot(edge);
  const double at_start = tangent.dot(geometry.incident_start - geometry.face_start);
  const double at_end = tangent.dot(geometry.incident_start - geometry.face_end);

  double low = 0.0;
  double high = 1.0;
  ClippedPoint low_point;
  ClippedPoint high_point;

  if (slope > 0.0) {
    if (-at_start / slope > low) {
      low = -at_start / slope;
      low_point = {low, true, geometry.face_start};
    }
    if (-at_end / slope < high) {
      high = -at_end / slope;
      high_point = {high, true, geometry.face_end};
    }
  } else if (slope < 0.0) {
    if (-at_start / slope < high) {
      high = -at_start / slope;
      high_point = {high, true, geometry.face_start};
    }
    if (-at_end / slope > low) {
      low = -at_end / slope;
      low_point = {low, true, geometry.face_end};
    }
  } else if (at_start < 0.0 || at_end > 0.0) {
    // Incident edge perpendicular to the face, so neither constraint
    // involves t: the whole edge is either inside the extent or outside.
    return geometry;
  }

  if (low > high) {
    return geometry;  // the edge misses the face's extent entirely
  }

  low_point.parameter = low;
  high_point.parameter = high;
  geometry.points.push_back(low_point);
  if (high > low) {
    geometry.points.push_back(high_point);
  }
  return geometry;
}

// ---------------------------------------------------------------------
// Second-order jets over the six degrees of freedom [first | second].
//
// The pair gap is a composition of a handful of primitives -- a normal
// that rotates with one body, a vertex that is material in the other, a
// clip parameter that is a ratio of two dot products -- and its Hessian
// is what the contact force's position Jacobian needs.
//
// Hand-assembling twenty-one Hessian entries through that composition is
// possible and would be very hard to trust. Carrying value, gradient and
// Hessian together through each operation instead makes the chain rule
// structural: every entry is produced by the same three product rules,
// and the gradient comes out as a free cross-check against the Jacobian
// that was derived independently.
//
// This is exact closed-form differentiation, not numerical: no step
// size, no truncation. File-local, used by one calculation.
// ---------------------------------------------------------------------

using Matrix6d = Eigen::Matrix<double, 6, 6>;


struct VectorJet {
  Eigen::Vector2d value = Eigen::Vector2d::Zero();
  Eigen::Matrix<double, 2, 6> gradient = Eigen::Matrix<double, 2, 6>::Zero();
  Matrix6d hessian[2] = {Matrix6d::Zero(), Matrix6d::Zero()};
};


struct ScalarJet {
  double value = 0.0;
  Eigen::Matrix<double, 1, 6> gradient = Eigen::Matrix<double, 1, 6>::Zero();
  Matrix6d hessian = Matrix6d::Zero();
};


ScalarJet Constant(double value) {
  ScalarJet jet;
  jet.value = value;
  return jet;
}


// A vector that rotates rigidly with one body and ignores its
// translation: a face normal, a tangent, an edge. Rotating by dtheta
// gives v^perp, and again gives -v.
VectorJet RotatingVector(const Eigen::Vector2d& value, int theta) {
  VectorJet jet;
  jet.value = value;
  const Eigen::Vector2d swept = Perp(value);
  jet.gradient(0, theta) = swept.x();
  jet.gradient(1, theta) = swept.y();
  jet.hessian[0](theta, theta) = -value.x();
  jet.hessian[1](theta, theta) = -value.y();
  return jet;
}


// A point fixed in one body: it follows the translation and sweeps
// (p - c)^perp under rotation.
VectorJet MaterialPoint(const Eigen::Vector2d& point, const Eigen::Vector2d& center, int offset) {
  VectorJet jet;
  jet.value = point;
  jet.gradient(0, offset) = 1.0;
  jet.gradient(1, offset + 1) = 1.0;
  const Eigen::Vector2d arm = point - center;
  const Eigen::Vector2d swept = Perp(arm);
  jet.gradient(0, offset + 2) = swept.x();
  jet.gradient(1, offset + 2) = swept.y();
  jet.hessian[0](offset + 2, offset + 2) = -arm.x();
  jet.hessian[1](offset + 2, offset + 2) = -arm.y();
  return jet;
}


VectorJet Subtract(const VectorJet& left, const VectorJet& right) {
  VectorJet jet;
  jet.value = left.value - right.value;
  jet.gradient = left.gradient - right.gradient;
  jet.hessian[0] = left.hessian[0] - right.hessian[0];
  jet.hessian[1] = left.hessian[1] - right.hessian[1];
  return jet;
}


VectorJet Add(const VectorJet& left, const VectorJet& right) {
  VectorJet jet;
  jet.value = left.value + right.value;
  jet.gradient = left.gradient + right.gradient;
  jet.hessian[0] = left.hessian[0] + right.hessian[0];
  jet.hessian[1] = left.hessian[1] + right.hessian[1];
  return jet;
}


// f = sum_k a_k b_k, so H_f = sum_k [ b_k H_{a_k} + a_k H_{b_k}
//                                   + grad a_k (x) grad b_k
//                                   + grad b_k (x) grad a_k ].
ScalarJet Dot(const VectorJet& left, const VectorJet& right) {
  ScalarJet jet;
  jet.value = left.value.dot(right.value);
  for (int k = 0; k < 2; ++k) {
    jet.gradient += left.value(k) * right.gradient.row(k) + right.value(k) * left.gradient.row(k);
    jet.hessian += right.value(k) * left.hessian[k] + left.value(k) * right.hessian[k];
    jet.hessian += left.gradient.row(k).transpose() * right.gradient.row(k);
    jet.hessian += right.gradient.row(k).transpose() * left.gradient.row(k);
  }
  return jet;
}


// V_k = s * u_k, the ordinary product rule twice over.
VectorJet Scale(const VectorJet& vector, const ScalarJet& scalar) {
  VectorJet jet;
  jet.value = scalar.value * vector.value;
  for (int k = 0; k < 2; ++k) {
    jet.gradient.row(k) = scalar.value * vector.gradient.row(k) + vector.value(k) * scalar.gradient;
    jet.hessian[k] = scalar.value * vector.hessian[k] + vector.value(k) * scalar.hessian;
    jet.hessian[k] += scalar.gradient.transpose() * vector.gradient.row(k);
    jet.hessian[k] += vector.gradient.row(k).transpose() * scalar.gradient;
  }
  return jet;
}


// From f*w = u, differentiated twice:
//   grad f = (grad u - f grad w) / w
//   H_f    = (H_u - f H_w - grad f (x) grad w - grad w (x) grad f) / w
ScalarJet Divide(const ScalarJet& numerator, const ScalarJet& denominator) {
  ScalarJet jet;
  jet.value = numerator.value / denominator.value;
  jet.gradient = (numerator.gradient - jet.value * denominator.gradient) / denominator.value;
  jet.hessian = numerator.hessian - jet.value * denominator.hessian;
  jet.hessian -= jet.gradient.transpose() * denominator.gradient;
  jet.hessian -= denominator.gradient.transpose() * jet.gradient;
  jet.hessian /= denominator.value;
  return jet;
}


ScalarJet Negate(const ScalarJet& scalar) {
  ScalarJet jet;
  jet.value = -scalar.value;
  jet.gradient = -scalar.gradient;
  jet.hessian = -scalar.hessian;
  return jet;
}


// The gap, built from primitives, carrying both derivatives. Material
// vertices and clipped points differ only in whether the edge parameter
// is a constant or a ratio -- everything downstream is identical.
ScalarJet GapJet(const PairGeometry& geometry, const ClippedPoint& clipped, const Eigen::Vector2d& reference_center, const Eigen::Vector2d& incident_center) {
  const int reference_offset = geometry.reference_is_first ? 0 : 3;
  const int incident_offset = geometry.reference_is_first ? 3 : 0;

  const VectorJet normal = RotatingVector(geometry.normal, reference_offset + 2);
  const VectorJet face_start = MaterialPoint(geometry.face_start, reference_center, reference_offset);
  const VectorJet incident_start = MaterialPoint(geometry.incident_start, incident_center, incident_offset);
  const VectorJet edge = RotatingVector(geometry.incident_end - geometry.incident_start, incident_offset + 2);

  ScalarJet parameter = Constant(clipped.parameter);
  if (clipped.clipped) {
    const VectorJet tangent = RotatingVector((geometry.face_end - geometry.face_start).normalized(), reference_offset + 2);
    const VectorJet clip_origin = MaterialPoint(clipped.clip_origin, reference_center, reference_offset);
    parameter = Negate(Divide(Dot(tangent, Subtract(incident_start, clip_origin)), Dot(tangent, edge)));
  }

  const VectorJet point = Add(incident_start, Scale(edge, parameter));
  return Dot(normal, Subtract(point, face_start));
}

}  // namespace

std::vector<Contact> detect_contacts_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Vector2d center = state.q.head<2>();
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();

  std::vector<Contact> contacts(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    const Eigen::Vector2d world_point = center + rotation * shape.vertices[i];
    contacts[i].vertex_index = static_cast<int>(i);
    contacts[i].signed_distance = plane.normal.dot(world_point) - plane.offset;
    contacts[i].normal = plane.normal;
    contacts[i].point = world_point;
  }
  return contacts;
}


std::vector<Eigen::RowVector3d> detect_contacts_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();

  std::vector<Eigen::RowVector3d> jacobians(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    // arm = p_i - c = R(theta) * vertices[i]; no need to add the center
    // back only to subtract it again.
    const Eigen::Vector2d arm = rotation * shape.vertices[i];
    jacobians[i] << plane.normal.x(), plane.normal.y(), plane.normal.dot(Perp(arm));
  }
  return jacobians;
}


std::vector<Eigen::RowVector3d> detect_contacts_body_perp_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();
  const Eigen::Vector2d tangent = Perp(plane.normal);

  std::vector<Eigen::RowVector3d> jacobians(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    // Identical construction to the normal row, with n^perp in place of
    // n -- the perp operator appears twice for different reasons: once
    // to turn the plane normal into the surface direction, once because
    // a vertex's velocity under unit angular rate is (p_i - c)^perp.
    const Eigen::Vector2d arm = rotation * shape.vertices[i];
    jacobians[i] << tangent.x(), tangent.y(), tangent.dot(Perp(arm));
  }
  return jacobians;
}


std::vector<PairContact> detect_contacts_pair(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape) {
  const PairGeometry geometry = ComputePairGeometry(first, first_shape, second, second_shape);
  const Eigen::Vector2d edge = geometry.incident_end - geometry.incident_start;

  std::vector<PairContact> contacts;
  contacts.reserve(geometry.points.size());
  for (const ClippedPoint& clipped : geometry.points) {
    PairContact contact;
    contact.point = geometry.incident_start + clipped.parameter * edge;
    contact.signed_distance = geometry.normal.dot(contact.point - geometry.face_start);
    // Normalized to point first -> second whichever body owns the face.
    contact.normal = geometry.reference_is_first ? geometry.normal : Eigen::Vector2d(-geometry.normal);
    contact.reference_is_first = geometry.reference_is_first;
    contact.clipped = clipped.clipped;
    contacts.push_back(contact);
  }
  return contacts;
}


std::vector<PairJacobian> detect_contacts_pair_jacobian(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape) {
  const PairGeometry geometry = ComputePairGeometry(first, first_shape, second, second_shape);
  const Eigen::Vector2d reference_center = (geometry.reference_is_first ? first : second).q.head<2>();
  const Eigen::Vector2d incident_center = (geometry.reference_is_first ? second : first).q.head<2>();

  std::vector<PairJacobian> jacobians(geometry.points.size());
  for (std::size_t k = 0; k < geometry.points.size(); ++k) {
    jacobians[k] = GapJet(geometry, geometry.points[k], reference_center, incident_center).gradient;
  }
  return jacobians;
}


std::vector<PairHessian> detect_contacts_pair_hessian(const RigidBodyState& first, const BodyShape& first_shape, const RigidBodyState& second, const BodyShape& second_shape) {
  const PairGeometry geometry = ComputePairGeometry(first, first_shape, second, second_shape);
  const Eigen::Vector2d reference_center = (geometry.reference_is_first ? first : second).q.head<2>();
  const Eigen::Vector2d incident_center = (geometry.reference_is_first ? second : first).q.head<2>();

  std::vector<PairHessian> hessians(geometry.points.size());
  for (std::size_t k = 0; k < geometry.points.size(); ++k) {
    hessians[k] = GapJet(geometry, geometry.points[k], reference_center, incident_center).hessian;
  }
  return hessians;
}

}  // namespace grip
