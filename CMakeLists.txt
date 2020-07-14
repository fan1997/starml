cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(starml VERSION 0.0.1 LANGUAGES CXX)

option(STARML_BUILD_SHARED_LIBS "Compiled as a shared library" ON)
option(STARML_ENABLE_TESTING "Build with testing enabled" ON)
option(STARML_USE_CUDA "Build with CUDA supported" ON)

# Show the detail of compile messages
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_COLOR_MAKEFILE   ON)
include(cmake/public/Utils.cmake)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(STARML_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STARML_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(STARML_EXTERNAL_DIR ${AITISA_API_ROOT_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Find all the files under `starml/`
file(GLOB_RECURSE starml_srcs "starml/*.cpp" "starml/*.cu")
if(NOT STARML_USE_CUDA)
  file(GLOB_RECURSE cuda_srcs "starml/*_cuda.cpp" "starml/*.cu")
  exclude(starml_srcs "${starml_srcs}" ${cuda_srcs})
endif()

# Create library target with all the source codes.
if(STARML_BUILD_SHARED_LIBS)
  add_library(starml SHARED ${starml_srcs})
else()
  add_library(starml STATIC ${starml_srcs})
endif()

target_include_directories(starml
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
# include_directories(${PROJECT_SOURCE_DIR}/include)



if(STARML_USE_CUDA)
  set_target_properties(starml PROPERTIES
    CUDA_SEPERABLE_COMPILATION ON
    INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

# Add dependency library link
set(starml_link_libraries)
if(STARML_USE_CUDA)
  include(cmake/public/CUDA.cmake)
  list(APPEND starml_link_libraries starml::cudart starml::cublas starml::thrust)
endif()
find_package(OpenMP REQUIRED QUIET)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(Eigen3 3.3 NO_MODULE)
if (Eigen3_FOUND)
    message("using external Eigen3")
    include_directories(${EIGEN3_INCLUDE_DIR})
    list(APPEND starml_link_libraries Eigen3::Eigen)
else ()
    message("using internal Eigen3 for acceleration")
    include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
endif ()

target_link_libraries(starml PUBLIC ${starml_link_libraries})

# Add testing
if(STARML_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(third_party/googletest)
  add_subdirectory(test)
endif()
