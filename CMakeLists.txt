cmake_minimum_required(VERSION 3.21)

project(crave LANGUAGES C CXX)

add_library(crave INTERFACE)
target_include_directories(crave INTERFACE src)

if(UNIX)
  target_link_libraries(crave INTERFACE m)
endif()
