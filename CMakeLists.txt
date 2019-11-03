cmake_minimum_required(VERSION 3.9)
project(minecrack LANGUAGES CXX CUDA)

if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    message(WARNING "Only the GCC compiler has been tested under Linux.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)


#libraries and paths

include_directories(cub/)

add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)
find_package(Boost REQUIRED program_options)

#we use the host compiler directly so we need to extract the cuda include path
get_filename_component(CUDA_ROOT ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_ROOT ${CUDA_ROOT} DIRECTORY)
include_directories(${CUDA_ROOT}/include)

#debug/optimization
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

#link/build flags
add_executable(minecrack 48bit/main.cpp 48bit/kernels.cu)
set_property(TARGET minecrack PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(minecrack Boost::program_options)
