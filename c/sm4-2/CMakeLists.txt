cmake_minimum_required(VERSION 3.10)
project(gm_test)

set(CMAKE_CXX_STANDARD 14)
find_package(Threads REQUIRED)
add_executable(gm_test gm_test.c gm.c randombytes.c sm2.c sm3.c sm4.c)

target_link_libraries(gm_test Threads::Threads)
