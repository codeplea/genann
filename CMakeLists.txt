cmake_minimum_required(VERSION 3.5)
project(genann) 
enable_testing()

IF(UNIX)
SET(M m)
#LSE
SET(M "")
ENDIF()

add_executable(example1 genann.c example1.c)
target_link_libraries(example1 ${M})
add_test(example1 example1 WORKING_DIRECTORY ..)

add_executable(example2 genann.c example2.c)
target_link_libraries(example2 ${M})
add_test(example2 example2 WORKING_DIRECTORY ..)

add_executable(example3 genann.c example3.c)
target_link_libraries(example3 ${M})
add_test(example3 example3 WORKING_DIRECTORY ..)

add_executable(example4 genann.c example4.c)
target_link_libraries(example4 ${M})
add_test(example4 example4 WORKING_DIRECTORY ..)

