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
  src/ode/TT06Model.cpp
  src/ode/PassiveModel.cpp
  src/ode/StewartPurkinjeModel.cpp
  src/ode/Grandi2011Model.cpp
  src/ode/RegionalIonicModel.cpp
  src/space/FiberTensorCoefficient.cpp
  src/space/Assembler.cpp
  src/solver/LinearSolverFactory.cpp
  src/solver/MonodomainStepper.cpp
  src/solver/PurkinjeCableSolver.cpp
  src/solver/PvjCoupler.cpp
  src/solver/ExtracellularRecoverySolver.cpp
  src/solver/TorsoPotentialSolver.cpp
  src/coupling/InterfaceMapper.cpp
  src/io/OutputManager.cpp
  src/io/CheckpointIO.cpp
  src/io/PseudoEcg.cpp
  src/ode/tt06_generated.c
  src/ode/stewart_generated.c
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

add_executable(generate_half_ellipsoid_case tools/generate_half_ellipsoid_case.cpp)
target_link_libraries(generate_half_ellipsoid_case PRIVATE monodomain_core)

add_executable(finalize_half_ellipsoid_mesh tools/finalize_half_ellipsoid_mesh.cpp)
target_link_libraries(finalize_half_ellipsoid_mesh PRIVATE monodomain_core)

add_executable(debug_stewart_currents tools/debug_stewart_currents.cpp)
target_link_libraries(debug_stewart_currents PRIVATE monodomain_core)

add_executable(test_tt06_single_cell tests/test_tt06_single_cell.cpp)
target_link_libraries(test_tt06_single_cell PRIVATE monodomain_core)
add_test(NAME tt06_single_cell COMMAND test_tt06_single_cell)

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

add_executable(test_stewart_single_cell tests/test_stewart_single_cell.cpp)
target_link_libraries(test_stewart_single_cell PRIVATE monodomain_core)
add_test(NAME stewart_single_cell COMMAND test_stewart_single_cell)

add_executable(test_purkinje_cable_smoke tests/test_purkinje_cable_smoke.cpp)
target_link_libraries(test_purkinje_cable_smoke PRIVATE monodomain_core)
add_test(NAME purkinje_cable_smoke COMMAND test_purkinje_cable_smoke)

add_executable(test_purkinje_cable_cv tests/test_purkinje_cable_cv.cpp)
target_link_libraries(test_purkinje_cable_cv PRIVATE monodomain_core)
add_test(NAME purkinje_cable_cv COMMAND test_purkinje_cable_cv)

add_executable(test_pvj_active_sign tests/test_pvj_active_sign.cpp)
target_link_libraries(test_pvj_active_sign PRIVATE monodomain_core)
add_test(NAME pvj_active_sign COMMAND test_pvj_active_sign)

add_executable(test_regional_isolation tests/test_regional_isolation.cpp)
target_link_libraries(test_regional_isolation PRIVATE monodomain_core)
add_test(NAME regional_isolation COMMAND test_regional_isolation)

add_executable(test_checkpoint_model_id tests/test_checkpoint_model_id.cpp)
target_link_libraries(test_checkpoint_model_id PRIVATE monodomain_core)
add_test(NAME checkpoint_model_id COMMAND test_checkpoint_model_id)
