cmake_minimum_required(VERSION 3.20)
project(grip CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  # /fp:precise is the MSVC default; set it explicitly since strict IEEE
  # determinism is a project requirement and /fp:fast must never sneak in.
  add_compile_options(/W4 /fp:precise)
endif()

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

# Off by default would be the right choice once anything links this as a
# library -- a physics dependency should not drag in a window toolkit.
# On for now because there are no consumers yet and the demo is the point.
option(GRIP_BUILD_DEMOS "Build the raylib demo (fetches raylib)" ON)
if(GRIP_BUILD_DEMOS)
  add_subdirectory(demos)
endif()
