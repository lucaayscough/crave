cmake_minimum_required(VERSION 3.21)

project(crave LANGUAGES C CXX)

add_library(crave INTERFACE)
target_include_directories(crave INTERFACE src)

if(UNIX)
  target_link_libraries(crave INTERFACE m)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC crave)
