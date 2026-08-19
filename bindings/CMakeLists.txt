include(FetchContent)

# Modern FindPython rather than pybind11's legacy default. The legacy path
# ignores -DPython_EXECUTABLE and exports the uppercase PYTHON_EXECUTABLE
# instead, so on a machine with more than one interpreter it silently builds
# against whichever it finds first -- and the test below would be handed an
# empty command.
set(PYBIND11_FINDPYTHON ON)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(grip_python grip_module.cpp)
set_target_properties(grip_python PROPERTIES OUTPUT_NAME grip)
target_link_libraries(grip_python PRIVATE grip_core)

# Exercises the module through Python rather than through GoogleTest, because
# the conversions and shape checks are the only thing here the C++ suite
# cannot reach.
add_test(NAME python_bindings COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bindings.py)
set_tests_properties(python_bindings PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:grip_python>")
