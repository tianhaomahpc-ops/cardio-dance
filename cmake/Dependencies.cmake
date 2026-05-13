find_package(MPI REQUIRED)
find_package(MFEM REQUIRED)

# Some MFEM installs (Spack with our share/mfem/config.mk path) only expose the
# raw `mfem` target; others expose `mfem::mfem`. Promote and alias to a single
# name so the rest of CMakeLists.txt only references `mfem::mfem`.
if(TARGET mfem AND NOT TARGET mfem::mfem)
  set_target_properties(mfem PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(mfem::mfem ALIAS mfem)
endif()

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
