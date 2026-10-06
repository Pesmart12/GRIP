add_library(grip_core
  api/simulate.cpp
  contact/detection.cpp
  contact/penalty.cpp
  dynamics/forces.cpp
  dynamics/integrator.cpp
  gradient/rollout.cpp
)

target_include_directories(grip_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grip_core PUBLIC Eigen3::Eigen)

# Compile position-independent code, so this static library can be linked
# into the shared Python module. ELF linkers reject non-PIC objects in a
# shared object outright; Windows has no such distinction, which is why the
# bindings built there and failed on the first Linux build.
set_target_properties(grip_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
