cmake_minimum_required(VERSION 3.9)
project(minecrack LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(cubiomes)

set(Boost_USE_STATIC_LIBS ON)
add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)
find_package(Boost REQUIRED program_options)

if(CMAKE_BUILD_TYPE MATCHES Release)
    find_package(OpenMP)
endif()

add_executable(minecrack-slime slime/main.cpp slime/slime_logic.cpp)
target_link_options(minecrack-slime PRIVATE -static)
target_link_libraries(minecrack-slime Boost::program_options)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minecrack-slime OpenMP::OpenMP_CXX)
endif()
