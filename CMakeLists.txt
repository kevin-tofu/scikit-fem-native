cmake_minimum_required(VERSION 3.18)
project(skfem_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  _skfn
  cpp/src/bindings.cpp
  cpp/src/runtime.cpp
  cpp/src/assembly/linear_form_assembler.cpp
  cpp/src/assembly/functional_assembler.cpp
  cpp/src/assembly/supermesh_builder.cpp
  cpp/src/assembly/bilinear_form_assembler.cpp
  cpp/src/assembly/cross_bilinear_assembler.cpp
  cpp/src/assembly/cut_form_assembler.cpp
  cpp/src/assembly/cut_cross_assembler.cpp
  cpp/src/assembly/basis_geometry.cpp
)
target_compile_features(_skfn PRIVATE cxx_std_17)
target_include_directories(_skfn PRIVATE cpp/include)
if(MSVC)
  target_compile_options(_skfn PRIVATE /O2 /W4)
else()
  target_compile_options(_skfn PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _skfn DESTINATION skfemntv)
