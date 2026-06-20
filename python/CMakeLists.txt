# pybind11 Python module `vordyn` -- the first Python surface for vorflow.
# Configure the project with -DVORONOI_BUILD_PYTHON=ON (pybind11 is fetched automatically).

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(vordyn bindings.cpp)
target_link_libraries(vordyn PRIVATE vorflow::vorflow)
target_compile_features(vordyn PRIVATE cxx_std_17)

# Smoke test (needs numpy in the interpreter). Runs the demo with PYTHONPATH pointed at the module.
if(VORONOI_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME test_vordyn_python
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_vordyn.py)
    set_tests_properties(test_vordyn_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:vordyn>")
  endif()
endif()
