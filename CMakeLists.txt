# Based on franzflasch
cmake_minimum_required(VERSION 2.8.4)
project (genann)
enable_testing()

IF(CMAKE_HOST_UNIX)
SET(M, m)
ELSE()
SET(M, "")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
ENDIF()

add_library(genann genann.c)
target_link_libraries(genann ${M})

add_executable(genann_test genann_test.c)
target_link_libraries(genann_test genann)
add_test(genann_test genann_test WORKING_DIRECTORY ..)

OPTION(WITH_EXAMPLES "Also build examples" OFF) # Enabled by default

IF(WITH_EXAMPLES)
    add_executable(example1 example1.c)
    target_link_libraries(example1 genann)
    add_executable(example2 example2.c)
    target_link_libraries(example2 genann)
    add_executable(example3 example3.c)
    target_link_libraries(example3 genann)
    add_executable(example4 example4.c)
    target_link_libraries(example4 genann)
ENDIF(WITH_EXAMPLES)

install (TARGETS genann
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)

### Debian Package generation
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "https://github.com/codeplea/genann")
include(CPack)
