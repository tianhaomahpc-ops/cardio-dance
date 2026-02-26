find_package(MPI REQUIRED)
find_package(MFEM REQUIRED)

# PETSc is optional at configure-time; runtime switch remains in config.
find_package(PETSc QUIET)
if(PETSc_FOUND)
  if(TARGET PETSC::petsc)
    message(STATUS "Found PETSc target PETSC::petsc")
  else()
    message(STATUS "PETSc found but imported target PETSC::petsc missing")
  endif()
else()
  message(STATUS "PETSc not found, building Hypre-only path")
endif()
