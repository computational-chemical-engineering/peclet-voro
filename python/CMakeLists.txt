# pybind11 Python module `vorflow` -- the Python surface for the library.
# Configure the project with -DVORONOI_BUILD_PYTHON=ON (pybind11 is fetched automatically).
# The CMake target is `vorflow_py` (the bare name `vorflow` is the C++ INTERFACE library);
# its output module is `vorflow`, imported from Python as `import vorflow`.

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(vorflow_py bindings.cpp)
set_target_properties(vorflow_py PROPERTIES OUTPUT_NAME vorflow)
target_link_libraries(vorflow_py PRIVATE vorflow::vorflow)
target_compile_features(vorflow_py PRIVATE cxx_std_17)

# Smoke test (needs numpy in the interpreter). Runs the demo with PYTHONPATH pointed at the module.
if(VORONOI_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME test_vorflow_python
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_vorflow.py)
    set_tests_properties(test_vorflow_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:vorflow_py>")
  endif()
endif()
