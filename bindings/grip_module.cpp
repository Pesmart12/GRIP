// Python bindings for the batched API.
//
// Deliberately mechanical. Everything with any judgement in it lives in
// src/api/, where the GoogleTest suite can reach it -- logic that lands here
// is logic the C++ tests cannot see, so there is as little of it as possible.
// What remains is shape checking, the numpy conversions, and docstrings.
//
// The array layouts are src/api/scene.hpp's, unchanged. numpy reshapes into
// them rather than translating, because that layout is Pack's ordering viewed
// as an array -- which is the whole reason it was chosen.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "api/scene.hpp"
#include "api/simulate.hpp"

namespace py = pybind11;

namespace {

using grip::ControlBatch;
using grip::RolloutGradientBatch;
using grip::Scene;
using grip::StateBatch;
using grip::TrajectoryBatch;

// forcecast so a caller passing float32, or a non-contiguous view, gets a
// converted copy rather than an error. Every array arriving here is read and
// then finished with, so the copy costs nothing that matters -- the one array
// big enough to care about goes the other way, and that one is not copied.
using InputArray = py::array_t<double, py::array::c_style | py::array::forcecast>;


std::string DescribeShape(const InputArray& array) {
  std::string text = "(";
  for (py::ssize_t i = 0; i < array.ndim(); ++i) {
    text += std::to_string(array.shape(i));
    text += (i + 1 < array.ndim()) ? ", " : "";
  }
  return text + ")";
}


void RequireRank(const InputArray& array, py::ssize_t rank, py::ssize_t trailing, const char* what, const char* expected) {
  if (array.ndim() != rank || array.shape(rank - 1) != trailing) {
    throw std::invalid_argument(std::string(what) + " has shape " + DescribeShape(array) + ", expected " + expected);
  }
}


StateBatch ToStateBatch(const InputArray& array, const char* what) {
  RequireRank(array, 3, 6, what, "(environments, bodies, 6)");
  StateBatch batch = grip::make_state_batch(static_cast<std::size_t>(array.shape(0)), static_cast<std::size_t>(array.shape(1)));
  const double* source = array.data();
  batch.values.assign(source, source + batch.values.size());
  return batch;
}


ControlBatch ToControlBatch(const InputArray& array, const char* what) {
  RequireRank(array, 4, 3, what, "(steps, environments, bodies, 3)");
  ControlBatch batch = grip::make_control_batch(static_cast<std::size_t>(array.shape(0)), static_cast<std::size_t>(array.shape(1)), static_cast<std::size_t>(array.shape(2)));
  const double* source = array.data();
  batch.values.assign(source, source + batch.values.size());
  return batch;
}


TrajectoryBatch ToTrajectoryBatch(const InputArray& array, const char* what) {
  RequireRank(array, 4, 6, what, "(steps + 1, environments, bodies, 6)");
  if (array.shape(0) < 1) {
    throw std::invalid_argument(std::string(what) + " must have at least one recorded state");
  }
  TrajectoryBatch batch = grip::make_trajectory_batch(static_cast<std::size_t>(array.shape(0)) - 1, static_cast<std::size_t>(array.shape(1)), static_cast<std::size_t>(array.shape(2)));
  const double* source = array.data();
  batch.values.assign(source, source + batch.values.size());
  return batch;
}


// Hands the buffer to numpy without copying it: the batch is moved to the
// heap and a capsule made responsible for deleting it, so the array owns the
// memory it is viewing and outlives this call.
//
// This is the one place zero-copy matters. A trajectory is tens of megabytes
// at the horizons a first-order learner uses, which is why TrajectoryBatch is
// contiguous in the first place.
template <typename Batch>
py::array_t<double> ReleaseToNumpy(Batch&& batch, std::vector<py::ssize_t> shape) {
  auto* held = new Batch(std::move(batch));
  py::capsule owner(held, [](void* raw) { delete static_cast<Batch*>(raw); });
  return py::array_t<double>(std::move(shape), held->values.data(), owner);
}


py::array_t<double> StepBatch(const std::vector<Scene>& scenes, const InputArray& state, const InputArray& controls, std::size_t substeps) {
  StateBatch batch = ToStateBatch(state, "state");
  grip::step_batch(scenes, batch, ToControlBatch(controls, "controls"), substeps);

  const auto environments = static_cast<py::ssize_t>(batch.environments);
  const auto bodies = static_cast<py::ssize_t>(batch.bodies);
  return ReleaseToNumpy(std::move(batch), {environments, bodies, 6});
}


py::array_t<double> RolloutBatch(const std::vector<Scene>& scenes, const InputArray& initial, const InputArray& controls, std::size_t substeps) {
  TrajectoryBatch trajectory;
  grip::rollout_batch(scenes, ToStateBatch(initial, "initial"), ToControlBatch(controls, "controls"), substeps, trajectory);

  const auto steps = static_cast<py::ssize_t>(trajectory.steps);
  const auto environments = static_cast<py::ssize_t>(trajectory.environments);
  const auto bodies = static_cast<py::ssize_t>(trajectory.bodies);
  return ReleaseToNumpy(std::move(trajectory), {steps + 1, environments, bodies, 6});
}


py::tuple AdjointBatch(const std::vector<Scene>& scenes, const InputArray& trajectory, const InputArray& controls, std::size_t substeps, const InputArray& dl_dZ, const InputArray& dl_dU) {
  RolloutGradientBatch gradients = grip::adjoint_batch(scenes, ToTrajectoryBatch(trajectory, "trajectory"), ToControlBatch(controls, "controls"), substeps, ToTrajectoryBatch(dl_dZ, "dl_dZ"), ToControlBatch(dl_dU, "dl_dU"));

  const auto steps = static_cast<py::ssize_t>(gradients.dJ_dU.steps);
  const auto environments = static_cast<py::ssize_t>(gradients.dJ_dZ0.environments);
  const auto bodies = static_cast<py::ssize_t>(gradients.dJ_dZ0.bodies);
  py::array_t<double> initial_state_gradient = ReleaseToNumpy(std::move(gradients.dJ_dZ0), {environments, bodies, 6});
  py::array_t<double> control_gradient = ReleaseToNumpy(std::move(gradients.dJ_dU), {steps, environments, bodies, 3});
  return py::make_tuple(initial_state_gradient, control_gradient);
}

}  // namespace

PYBIND11_MODULE(grip, module) {
  module.doc() = "GRIP -- a 2D differentiable rigid-body contact simulator.\n\nSimulates and differentiates. Cost functions, policies and training loops\nbelong to whatever calls this: supply the partial derivatives of your\nobjective as seeds, and total derivatives come back.";

  py::class_<grip::RigidBodyParams>(module, "RigidBodyParams", "Mass properties. Never differentiated with respect to.")
      .def(py::init<>())
      .def(py::init([](double mass, double inertia) { return grip::RigidBodyParams{mass, inertia}; }), py::arg("mass"), py::arg("inertia"))
      .def_readwrite("mass", &grip::RigidBodyParams::mass)
      .def_readwrite("inertia", &grip::RigidBodyParams::inertia, "Moment of inertia about the centre of mass.");

  py::class_<grip::BodyShape>(module, "BodyShape", "A convex polygon, vertices in the body frame relative to the centre of mass. Must wind counterclockwise.")
      .def(py::init<>())
      .def(py::init([](const std::vector<Eigen::Vector2d>& vertices) { return grip::BodyShape{vertices}; }), py::arg("vertices"))
      .def_readwrite("vertices", &grip::BodyShape::vertices);

  py::class_<grip::HalfPlane>(module, "HalfPlane", "Static scenery in Hesse normal form: free space is {p : normal.p >= offset}. The normal must be unit length -- signed distance is normal.p - offset, which is a true distance only then.")
      .def(py::init<>())
      .def(py::init([](const Eigen::Vector2d& normal, double offset) { return grip::HalfPlane{normal, offset}; }), py::arg("normal"), py::arg("offset") = 0.0)
      .def_readwrite("normal", &grip::HalfPlane::normal)
      .def_readwrite("offset", &grip::HalfPlane::offset);

  py::class_<grip::PenaltyParams>(module, "PenaltyParams", "Contact model parameters. Zero stiffness means no contact response, zero damping recovers the undamped spring, and zero friction is frictionless whatever slip_damping is.")
      .def(py::init<>())
      .def(py::init([](double stiffness, double damping, double slip_damping, double friction) { return grip::PenaltyParams{stiffness, damping, slip_damping, friction}; }), py::arg("stiffness") = 0.0, py::arg("damping") = 0.0, py::arg("slip_damping") = 0.0, py::arg("friction") = 0.0)
      .def_readwrite("stiffness", &grip::PenaltyParams::stiffness)
      .def_readwrite("damping", &grip::PenaltyParams::damping)
      .def_readwrite("slip_damping", &grip::PenaltyParams::slip_damping)
      .def_readwrite("friction", &grip::PenaltyParams::friction, "Coulomb mu. Its arctangent is the steepest slope a body rests on.");

  py::class_<Scene>(module, "Scene", "One environment's invariant data. Configuration, not a simulator: state is passed in and returned out, never held here.\n\nThe list attributes hand back copies, so mutate them by assigning a whole list rather than by appending in place.")
      .def(py::init<>())
      .def(py::init([](std::vector<grip::RigidBodyParams> params, std::vector<grip::BodyShape> shapes, grip::HalfPlane plane, grip::PenaltyParams penalty, double dt, double gravity) {
             Scene scene;
             scene.params = std::move(params);
             scene.shapes = std::move(shapes);
             scene.plane = plane;
             scene.penalty = penalty;
             scene.dt = dt;
             scene.gravity = gravity;
             return scene;
           }),
           py::arg("params"), py::arg("shapes"), py::arg("plane") = grip::HalfPlane{}, py::arg("penalty") = grip::PenaltyParams{}, py::arg("dt") = 1.0e-3, py::arg("gravity") = grip::kDefaultGravity)
      .def_readwrite("params", &Scene::params)
      .def_readwrite("shapes", &Scene::shapes)
      .def_readwrite("plane", &Scene::plane)
      .def_readwrite("penalty", &Scene::penalty)
      .def_readwrite("dt", &Scene::dt, "Integration timestep, not the control period. One control step is `substeps` of these.")
      .def_readwrite("gravity", &Scene::gravity);

  module.def("step_batch", &StepBatch, py::arg("scenes"), py::arg("state"), py::arg("controls"), py::arg("substeps") = 1,
             "Advance every environment by `substeps` integration steps under one held wrench.\n\nstate     (environments, bodies, 6), the last axis (x, y, theta, vx, vy, omega)\ncontrols  (1, environments, bodies, 3)\nreturns   (environments, bodies, 6)\n\nReturns a new array rather than writing in place, which is what a numpy caller expects and what keeps a float32 input from being silently updated into a discarded copy.");

  module.def("rollout_batch", &RolloutBatch, py::arg("scenes"), py::arg("initial"), py::arg("controls"), py::arg("substeps") = 1,
             "Roll every environment forward, recording one state per control step plus the initial one.\n\ninitial   (environments, bodies, 6)\ncontrols  (steps, environments, bodies, 3)\nreturns   (steps + 1, environments, bodies, 6)\n\nThe returned array views the simulator's own buffer without copying it.");

  module.def("adjoint_batch", &AdjointBatch, py::arg("scenes"), py::arg("trajectory"), py::arg("controls"), py::arg("substeps"), py::arg("dl_dZ"), py::arg("dl_dU"),
             "Reverse-mode sweep over a trajectory from rollout_batch.\n\nGRIP never sees your objective. Supply its partials and receive total derivatives.\n\ndl_dZ     (steps + 1, environments, bodies, 6), d(stage cost)/d(state)\ndl_dU     (steps, environments, bodies, 3), d(stage cost)/d(control)\nreturns   (dJ_dZ0, dJ_dU) shaped like the initial state and the controls\n\nA terminal-only objective is the case where every dl_dZ entry but the last is zero.");

  module.attr("__version__") = "0.1.0";
}
