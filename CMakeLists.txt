cmake_minimum_required(VERSION 3.20)
project(monodomain_tt06 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

add_library(monodomain_core
  src/config/SimulationConfig.cpp
  src/mechanics/ElasticitySolver.cpp
  src/ode/Grandi2011Model.cpp
  src/ode/PassiveModel.cpp
  src/ode/RegionalIonicModel.cpp
  src/ode/TT06Model.cpp
  src/ode/grandi2011_generated.c
  src/space/FiberTensorCoefficient.cpp
  src/space/Assembler.cpp
  src/solver/LinearSolverFactory.cpp
  src/solver/MonodomainStepper.cpp
  src/solver/ExtracellularRecoverySolver.cpp
  src/solver/TorsoPotentialSolver.cpp
  src/coupling/InterfaceMapper.cpp
  src/io/OutputManager.cpp
  src/io/CheckpointIO.cpp
  src/ode/tt06_generated.c
)

target_include_directories(monodomain_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(monodomain_core
  PUBLIC
    MPI::MPI_CXX
    mfem::mfem
)

if(TARGET PETSC::petsc)
  target_link_libraries(monodomain_core PUBLIC PETSC::petsc)
endif()

add_executable(monodomain src/main.cpp)
target_link_libraries(monodomain PRIVATE monodomain_core)

add_executable(generate_niederer_case tools/generate_niederer_case.cpp)
target_link_libraries(generate_niederer_case PRIVATE monodomain_core)

add_executable(generate_torso_box tools/generate_torso_box.cpp)
target_link_libraries(generate_torso_box PRIVATE monodomain_core)

add_executable(generate_conforming_wholebody_case tools/generate_conforming_wholebody_case.cpp)
target_link_libraries(generate_conforming_wholebody_case PRIVATE monodomain_core)

add_executable(generate_mesh_attr_gf tools/generate_mesh_attr_gf.cpp)
target_link_libraries(generate_mesh_attr_gf PRIVATE monodomain_core)

add_executable(split_niederer_x_regions tools/split_niederer_x_regions.cpp)
target_link_libraries(split_niederer_x_regions PRIVATE monodomain_core)

add_executable(generate_regional_fibers tools/generate_regional_fibers.cpp)
target_link_libraries(generate_regional_fibers PRIVATE monodomain_core)

add_executable(solve_elasticity tools/solve_elasticity.cpp)
target_link_libraries(solve_elasticity PRIVATE monodomain_core)

add_executable(test_tt06_single_cell tests/test_tt06_single_cell.cpp)
target_link_libraries(test_tt06_single_cell PRIVATE monodomain_core)
add_test(NAME tt06_single_cell COMMAND test_tt06_single_cell)

add_executable(test_passive_model tests/test_passive_model.cpp)
target_link_libraries(test_passive_model PRIVATE monodomain_core)
add_test(NAME passive_model COMMAND test_passive_model)

add_executable(test_config_parse tests/test_config_parse.cpp)
target_link_libraries(test_config_parse PRIVATE monodomain_core)
add_test(NAME config_parse COMMAND test_config_parse)

add_executable(test_fiber_gf_load tests/test_fiber_gf_load.cpp)
target_link_libraries(test_fiber_gf_load PRIVATE monodomain_core)
add_test(NAME fiber_gf_load COMMAND test_fiber_gf_load)

add_executable(test_restart_resume tests/test_restart_resume.cpp)
target_link_libraries(test_restart_resume PRIVATE monodomain_core)
add_test(NAME restart_resume COMMAND test_restart_resume)

add_executable(test_wholebody_smoke tests/test_wholebody_smoke.cpp)
target_link_libraries(test_wholebody_smoke PRIVATE monodomain_core)
add_test(NAME wholebody_smoke COMMAND test_wholebody_smoke)

add_executable(test_elasticity_monodomain_mesh tests/test_elasticity_monodomain_mesh.cpp)
target_link_libraries(test_elasticity_monodomain_mesh PRIVATE monodomain_core)
add_test(NAME elasticity_monodomain_mesh COMMAND test_elasticity_monodomain_mesh)
