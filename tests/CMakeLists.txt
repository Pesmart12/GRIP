include(GoogleTest)

add_executable(grip_unit_tests
  unit/test_contact_detection.cpp
  unit/test_forces.cpp
  unit/test_integrator.cpp
  unit/test_integrator_system.cpp
  unit/test_mass.cpp
  unit/test_penalty_force.cpp
  unit/test_rollout.cpp
)
target_link_libraries(grip_unit_tests PRIVATE grip_core GTest::gtest_main)
gtest_discover_tests(grip_unit_tests)

add_executable(grip_validation_tests
  validation/test_contact_boundary.cpp
  validation/test_contact_jacobians.cpp
  validation/test_energy_behavior.cpp
  validation/test_integrator_jacobians.cpp
  validation/test_integrator_system_jacobians.cpp
  validation/test_penalty_jacobians.cpp
  validation/test_rollout_gradients.cpp
)
target_include_directories(grip_validation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grip_validation_tests PRIVATE grip_core GTest::gtest_main)
gtest_discover_tests(grip_validation_tests)
