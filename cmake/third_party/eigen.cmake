include(FetchContent)

set(EIGEN_VERSION 3.3.7)
set(EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz)

set(FETCHCONTENT_BASE_DIR ${THIRD_PARTY_DIR}/eigen)
FetchContent_Populate(eigen
  URL ${EIGEN_URL} 
  SUBBUILD_DIR ${THIRD_PARTY_DIR}/eigen
  SOURCE_DIR ${THIRD_PARTY_DIR}/eigen/eigen-src
  BINARY_DIR ${THIRD_PARTY_DIR}/eigen/eigen-build
)
add_subdirectory(
  ${eigen_SOURCE_DIR}
  ${eigen_BINARY_DIR}
)
