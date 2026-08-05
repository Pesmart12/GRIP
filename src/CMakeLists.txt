add_library(grip_core
  contact/detection.cpp
  dynamics/forces.cpp
  dynamics/integrator.cpp
)

target_include_directories(grip_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grip_core PUBLIC Eigen3::Eigen)
