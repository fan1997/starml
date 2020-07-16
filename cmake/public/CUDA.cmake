include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(WARNING "starml: Cannot find CUDA, turn off the `USE_CUDA` option automatically.")
  set(STARML_USE_CUDA OFF)
  return()
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDA QUIET REQUIRED)

# add_library(starml::cuda INTERFACE IMPORTED)
# set_property(TARGET starml::cuda PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES})
# set_property(TARGET starml::cuda PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})
add_library(starml::cudart INTERFACE IMPORTED)
set_property(TARGET starml::cudart PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES})
set_property(TARGET starml::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

# cublas
add_library(starml::cublas INTERFACE IMPORTED)
message(${CUDA_CUBLAS_LIBRARIES})
if(STARML_BUILD_SHARED_LIBS)
    set_property(TARGET starml::cublas PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
else()
    set_property(TARGET starml::cublas PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
endif()
set_property(TARGET starml::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

# thrust is a Head-only library
add_library(starml::thrust INTERFACE IMPORTED)
set_property(TARGET starml::thrust PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-declarations")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# Allow Relocatable Device Code
list(APPEND CUDA_NVCC_FLAGS "-rdc=true")
STRING(REPLACE ";" " " CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
list(APPEND CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})
message(STATUS "Add CUDA NVCC flags: ${CUDA_NVCC_FLAGS}")
