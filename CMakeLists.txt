cmake_minimum_required(VERSION 3.5)
project(genann) 
enable_testing()

IF(CMAKE_HOST_UNIX)
SET(M m)
ELSE()
SET(M "")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
ENDIF()

add_executable(genann_test genann.c genann_test.c)
target_link_libraries(genann_test ${M})
add_test(genann_test genann_test WORKING_DIRECTORY ..)

add_executable(example1 genann.c example1.c)
target_link_libraries(example1 ${M})

add_executable(example2 genann.c example2.c)
target_link_libraries(example2 ${M})

add_executable(example3 genann.c example3.c)
target_link_libraries(example3 ${M})

add_executable(example4 genann.c example4.c)
target_link_libraries(example4 ${M})

