include(GoogleTest)

add_executable(grip_unit_tests
  unit/test_integrator.cpp
)
target_link_libraries(grip_unit_tests PRIVATE grip_core GTest::gtest_main)
gtest_discover_tests(grip_unit_tests)

add_executable(grip_validation_tests
  validation/test_energy_behavior.cpp
  validation/test_integrator_jacobians.cpp
)
target_include_directories(grip_validation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grip_validation_tests PRIVATE grip_core GTest::gtest_main)
gtest_discover_tests(grip_validation_tests)
