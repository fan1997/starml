function(starml_register_exe name srcs test)
    add_executable(${name} ${srcs})
    target_link_libraries(${name} ${ARGN})
    if("${test}" STREQUAL "test")
        add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
    endif()
endfunction()

list(APPEND starml_test_srcs 
  
)

# Build a exe for each test files
foreach(test_src ${starml_test_srcs})
  get_filename_component(test_name ${test_src} NAME_WE)
  starml_register_exe(${test_name} "${test_src}" "test" "starml" "starml::gmock")
endforeach()

# Build a exe including all test files 
set(starml_BUILD_TEST_MAIN ON)
if(${starml_BUILD_TEST_MAIN})
  add_executable(main main.cpp ${starml_test_srcs})
  target_link_libraries(main "starml" "starml::gmock")
endif()