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

# OpenMP parallelizes the batched API's loop over environments and nothing
# else. PUBLIC so tests, bindings and the benchmark compile with the same
# flags. Optional: without it the pragmas compile away and the loop is serial.
if(GRIP_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(grip_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# Compiled with OpenMP, Eigen may start parallelizing large matrix products
# on its own. Parallelism belongs at the environment level, where it cannot
# change any environment's arithmetic, so Eigen stays serial everywhere.
target_compile_definitions(grip_core PUBLIC EIGEN_DONT_PARALLELIZE)

# Compile position-independent code, so this static library can be linked
# into the shared Python module. ELF linkers reject non-PIC objects in a
# shared object outright; Windows has no such distinction, which is why the
# bindings built there and failed on the first Linux build.
set_target_properties(grip_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
