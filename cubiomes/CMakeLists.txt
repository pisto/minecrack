cmake_minimum_required(VERSION 3.9)
project(cubiomes C)

set(CMAKE_C_STANDARD 11)

add_library(cubiomes cubiomes/generator.c cubiomes/layers.c)
target_compile_options(cubiomes PRIVATE -fwrapv "$<$<CONFIG:RELEASE>:-O3>")
