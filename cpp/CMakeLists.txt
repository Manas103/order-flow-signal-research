cmake_minimum_required(VERSION 3.16)
project(obsr_incremental_features CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(ENABLE_ASAN "Build with Address/UndefinedBehavior sanitizers" OFF)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

include_directories(include)

add_executable(incremental_bench apps/incremental_bench.cpp)
add_executable(test_incremental tests/test_incremental.cpp)

enable_testing()
add_test(NAME incremental COMMAND test_incremental)
