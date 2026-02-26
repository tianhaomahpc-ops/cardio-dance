# Minimal MFEM finder for installs that provide share/mfem/config.mk only.
# Usage: set MFEM_DIR to installation prefix (or rely on env MFEM_DIR).

set(_mfem_hints "")
if(MFEM_DIR)
  list(APPEND _mfem_hints "${MFEM_DIR}")
endif()
if(DEFINED ENV{MFEM_DIR})
  list(APPEND _mfem_hints "$ENV{MFEM_DIR}")
endif()

find_path(MFEM_CONFIG_DIR
  NAMES config.mk
  PATH_SUFFIXES share/mfem
  HINTS ${_mfem_hints}
  DOC "MFEM configuration directory"
)

if(NOT MFEM_CONFIG_DIR)
  set(MFEM_FOUND FALSE)
  return()
endif()

get_filename_component(MFEM_PREFIX "${MFEM_CONFIG_DIR}/../.." ABSOLUTE)
set(MFEM_CONFIG_MK "${MFEM_CONFIG_DIR}/config.mk")
if(NOT EXISTS "${MFEM_CONFIG_MK}")
  set(MFEM_FOUND FALSE)
  return()
endif()

function(_mfem_extract _key _out)
  file(STRINGS "${MFEM_CONFIG_MK}" _line REGEX "^${_key}[ \\t]*=")
  if(_line)
    string(REGEX REPLACE "^${_key}[ \\t]*=[ \\t]*" "" _val "${_line}")
    string(STRIP "${_val}" _val)
    set(${_out} "${_val}" PARENT_SCOPE)
  else()
    set(${_out} "" PARENT_SCOPE)
  endif()
endfunction()

_mfem_extract("MFEM_INC_DIR" MFEM_INC_DIR)
_mfem_extract("MFEM_TPLFLAGS" MFEM_TPLFLAGS)
_mfem_extract("MFEM_EXT_LIBS" MFEM_EXT_LIBS)

if(NOT MFEM_INC_DIR)
  set(MFEM_INC_DIR "${MFEM_PREFIX}/include")
endif()

set(MFEM_LIB_FILE "${MFEM_PREFIX}/lib/libmfem.a")
if(NOT EXISTS "${MFEM_LIB_FILE}")
  set(MFEM_LIB_FILE "${MFEM_PREFIX}/lib/libmfem.dylib")
endif()

if(NOT EXISTS "${MFEM_LIB_FILE}")
  set(MFEM_FOUND FALSE)
  return()
endif()

set(_mfem_include_dirs "${MFEM_INC_DIR}")
if(MFEM_TPLFLAGS)
  separate_arguments(_mfem_tpl_tokens UNIX_COMMAND "${MFEM_TPLFLAGS}")
  foreach(tok IN LISTS _mfem_tpl_tokens)
    if(tok MATCHES "^-I(.+)")
      list(APPEND _mfem_include_dirs "${CMAKE_MATCH_1}")
    endif()
  endforeach()
endif()

set(_mfem_ext_tokens "")
if(MFEM_EXT_LIBS)
  separate_arguments(_mfem_ext_tokens UNIX_COMMAND "${MFEM_EXT_LIBS}")
endif()

if(NOT TARGET mfem)
  add_library(mfem INTERFACE IMPORTED)
endif()
if(NOT TARGET mfem::mfem)
  add_library(mfem::mfem ALIAS mfem)
endif()

target_include_directories(mfem INTERFACE ${_mfem_include_dirs})
target_link_libraries(mfem INTERFACE "${MFEM_LIB_FILE}" ${_mfem_ext_tokens})

set(MFEM_FOUND TRUE)
set(MFEM_INCLUDE_DIRS "${_mfem_include_dirs}")
set(MFEM_LIBRARIES "${MFEM_LIB_FILE};${_mfem_ext_tokens}")

mark_as_advanced(MFEM_PREFIX MFEM_CONFIG_DIR MFEM_CONFIG_MK MFEM_INC_DIR MFEM_LIB_FILE)
