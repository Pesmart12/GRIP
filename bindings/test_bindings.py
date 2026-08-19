"""Tests for the Python bindings.

Scoped to what the C++ suite cannot reach: array conversion, shape checking,
and that the layout numpy sees is the layout the physics uses. The physics
itself is tested in tests/ and is not re-tested here.

Plain asserts and a main, rather than pytest -- one test file does not justify
a dependency, and ctest reports the exit code either way.
"""

import sys

import numpy as np

import grip

UNIT_SQUARE = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]


def make_scene(bodies=1, dt=1.0e-3, stiffness=0.0):
    return grip.Scene(
        params=[grip.RigidBodyParams(mass=1.0, inertia=1.0 / 6.0) for _ in range(bodies)],
        shapes=[grip.BodyShape(UNIT_SQUARE) for _ in range(bodies)],
        plane=grip.HalfPlane(normal=[0.0, 1.0], offset=0.0),
        penalty=grip.PenaltyParams(stiffness=stiffness, damping=0.0),
        dt=dt,
    )


def test_free_fall_matches_the_closed_form():
    """Symplectic Euler from rest under gravity has an exact closed form.

    v_n = n*dt*a and q_n = dt^2*a*n(n+1)/2, which symplectic_euler.md derives.
    Checking against that rather than against another GRIP call means the
    binding is compared to arithmetic, not to itself.
    """
    steps, dt = 50, 1.0e-3
    scenes = [make_scene(dt=dt)]
    initial = np.zeros((1, 1, 6))
    initial[0, 0, 1] = 10.0
    controls = np.zeros((steps, 1, 1, 3))

    trajectory = grip.rollout_batch(scenes, initial, controls, substeps=1)
    assert trajectory.shape == (steps + 1, 1, 1, 6), trajectory.shape

    acceleration = -9.81
    for n in range(steps + 1):
        expected_v = n * dt * acceleration
        expected_y = 10.0 + dt * dt * acceleration * n * (n + 1) / 2.0
        assert np.isclose(trajectory[n, 0, 0, 4], expected_v, rtol=0, atol=1e-12), n
        assert np.isclose(trajectory[n, 0, 0, 1], expected_y, rtol=0, atol=1e-12), n


def test_last_axis_is_packs_ordering():
    """The layout claim, checked from the Python side.

    Position occupies 0:3 and velocity 3:6, so a caller reading
    state[..., 1] gets height and state[..., 4] gets vertical speed. If this
    ever silently changed, every gradient handed back would be scrambled in a
    way no shape check would catch.
    """
    scenes = [make_scene()]
    initial = np.zeros((1, 1, 6))
    initial[0, 0, 0:3] = [1.0, 2.0, 0.3]
    initial[0, 0, 3:6] = [4.0, 5.0, 0.6]
    controls = np.zeros((1, 1, 1, 3))

    trajectory = grip.rollout_batch(scenes, initial, controls, substeps=1)
    start = trajectory[0, 0, 0]
    assert np.array_equal(start, initial[0, 0]), (start, initial[0, 0])

    # One step of symplectic Euler: velocity updates first, then position uses
    # the new velocity. Horizontal motion is unforced, so it is exact.
    after = trajectory[1, 0, 0]
    assert np.isclose(after[3], 4.0), after
    assert np.isclose(after[0], 1.0 + 1.0e-3 * 4.0), after


def test_control_gradient_matches_the_worked_example():
    """adjoint.md's closed form for free flight.

    Seeding the final height gives dJ/du_t = dt^2 * (H - t) / m for the
    vertical component -- an early push has longer to act, so the earliest
    control has H times the influence of the last. Wrong sweep orientation
    reverses the ramp, which is the cheapest possible check that the
    backward pass runs the right way.
    """
    horizon, dt = 20, 1.0e-3
    scenes = [make_scene(dt=dt)]
    initial = np.zeros((1, 1, 6))
    initial[0, 0, 1] = 10.0
    controls = np.zeros((horizon, 1, 1, 3))

    trajectory = grip.rollout_batch(scenes, initial, controls, substeps=1)
    dl_dZ = np.zeros_like(trajectory)
    dl_dZ[horizon, 0, 0, 1] = 1.0
    dl_dU = np.zeros_like(controls)

    dJ_dZ0, dJ_dU = grip.adjoint_batch(scenes, trajectory, controls, 1, dl_dZ, dl_dU)
    assert dJ_dZ0.shape == (1, 1, 6), dJ_dZ0.shape
    assert dJ_dU.shape == (horizon, 1, 1, 3), dJ_dU.shape

    # dq_H/dv_0 = H*dt, the shear composing with itself.
    assert np.isclose(dJ_dZ0[0, 0, 4], horizon * dt), dJ_dZ0[0, 0]
    assert np.isclose(dJ_dZ0[0, 0, 1], 1.0), dJ_dZ0[0, 0]

    for t in range(horizon):
        assert np.isclose(dJ_dU[t, 0, 0, 1], dt * dt * (horizon - t)), t


def test_substeps_multiply_the_elapsed_time():
    """A control step of K substeps advances K integration steps.

    Rolling S steps with K substeps must land exactly where S*K steps with one
    substep land, since it is the same integrator applied the same number of
    times under the same (zero) wrench.
    """
    scenes = [make_scene(stiffness=0.0)]
    initial = np.zeros((1, 1, 6))
    initial[0, 0, 1] = 10.0

    coarse = grip.rollout_batch(scenes, initial, np.zeros((5, 1, 1, 3)), substeps=8)
    fine = grip.rollout_batch(scenes, initial, np.zeros((40, 1, 1, 3)), substeps=1)
    assert np.array_equal(coarse[-1], fine[-1]), (coarse[-1], fine[-1])


def test_environments_are_independent():
    """Different initial heights in one batch must not influence each other."""
    environments = 3
    scenes = [make_scene() for _ in range(environments)]
    initial = np.zeros((environments, 1, 6))
    initial[:, 0, 1] = [5.0, 10.0, 20.0]
    controls = np.zeros((7, environments, 1, 3))

    together = grip.rollout_batch(scenes, initial, controls, substeps=3)
    for environment in range(environments):
        alone = grip.rollout_batch([scenes[environment]], initial[environment : environment + 1], controls[:, environment : environment + 1], substeps=3)
        assert np.array_equal(together[:, environment : environment + 1], alone), environment


def test_scenes_may_differ_across_the_batch():
    """Only body count is rectangular. Gravity, plane and contact parameters
    are per environment, which is what a domain-randomized task needs."""
    scenes = [make_scene(), make_scene()]
    scenes[1].gravity = 1.62
    initial = np.zeros((2, 1, 6))
    initial[:, 0, 1] = 10.0

    trajectory = grip.rollout_batch(scenes, initial, np.zeros((10, 2, 1, 3)), substeps=1)
    assert trajectory[-1, 0, 0, 1] < trajectory[-1, 1, 0, 1], "stronger gravity must fall further"


def test_non_contiguous_and_float32_inputs_are_accepted():
    """forcecast earns its place: a caller slicing or using float32 gets a
    converted copy rather than an error."""
    scenes = [make_scene()]
    wide = np.zeros((1, 1, 12))
    view = wide[:, :, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    view[0, 0, 1] = 10.0

    from_view = grip.rollout_batch(scenes, view, np.zeros((3, 1, 1, 3)), substeps=1)
    from_float32 = grip.rollout_batch(scenes, np.ascontiguousarray(view).astype(np.float32), np.zeros((3, 1, 1, 3), dtype=np.float32), substeps=1)
    assert np.allclose(from_view, from_float32)


def test_malformed_shapes_raise():
    scenes = [make_scene()]
    initial = np.zeros((1, 1, 6))
    controls = np.zeros((4, 1, 1, 3))

    for bad, args in [
        ("rank", (scenes, np.zeros((1, 6)), controls)),
        ("trailing", (scenes, np.zeros((1, 1, 5)), controls)),
        ("environments", (scenes, np.zeros((2, 1, 6)), controls)),
        ("bodies", (scenes, initial, np.zeros((4, 1, 2, 3)))),
    ]:
        try:
            grip.rollout_batch(*args, substeps=1)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"expected a raise for {bad}")

    try:
        grip.rollout_batch(scenes, initial, controls, substeps=0)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("expected a raise for substeps=0")


def test_returned_trajectory_owns_its_memory():
    """The trajectory views the simulator's buffer through a capsule. If that
    ownership were wrong the memory would be freed while numpy still pointed
    at it, so read it back after the simulator call is long gone."""
    scenes = [make_scene()]
    initial = np.zeros((1, 1, 6))
    initial[0, 0, 1] = 10.0
    trajectory = grip.rollout_batch(scenes, initial, np.zeros((100, 1, 1, 3)), substeps=1)
    del scenes, initial

    copied = trajectory.copy()
    for _ in range(50):
        _ = np.zeros((1000, 100))  # churn the allocator
    assert np.array_equal(trajectory, copied)
    assert trajectory.base is not None, "expected a view over a capsule, not a copy"


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
