include(FetchContent)

set(GOOGLETEST_VERSION 1.8.0)
set(GOOGLETEST_URL https://github.com/google/googletest/archive/release-${GOOGLETEST_VERSION}.tar.gz)

set(FETCHCONTENT_BASE_DIR ${THIRD_PARTY_DIR}/googletest)
FetchContent_Populate(googletest
  URL ${GOOGLETEST_URL} 
  SUBBUILD_DIR ${THIRD_PARTY_DIR}/googletest 
  SOURCE_DIR ${THIRD_PARTY_DIR}/googletest/googletest-src
  BINARY_DIR ${THIRD_PARTY_DIR}/googletest/googletest-build
)

add_subdirectory(
  ${googletest_SOURCE_DIR}
  ${googletest_BINARY_DIR}
)
