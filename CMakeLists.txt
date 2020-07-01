cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(starml VERSION 0.0.1 LANGUAGES CXX)

############## Config Options ################
option(STARML_BUILD_SHARED_LIBS "Compiled as a shared library" OFF)
option(STARML_ENABLE_TESTING "Build with testing enabled" ON)
option(STARML_USE_CUDA "Build with CUDA supported" ON)

# Show the detail of compile messages
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_COLOR_MAKEFILE   ON)

set(STARML_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STARML_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(STARML_EXTERNAL_DIR ${AITISA_API_ROOT_DIR}/third_party)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

############### Target Create ##################
# Find all the files under `starml/`
file(GLOB_RECURSE starml_srcs "starml/*.cpp" "starml/*.cu")

# Create library target with all the source codes.
if(BUILD_SHARED_LIBS)
  add_library(starml SHARED ${starml_srcs})
else()
  add_library(starml STATIC ${starml_srcs})
endif()

# Add third party link
set(starml_link_libraries)
if(STARML_USE_CUDA)
  include(cmake/public/CUDA.cmake)
  list(APPEND starml_link_libraries starml::cudart starml::cublas starml::thrust)
endif()


if(STARML_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()