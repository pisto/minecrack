cmake_minimum_required(VERSION 3.9)
project(minecrack LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(cubiomes)
add_subdirectory(biome-java)

if(CMAKE_BUILD_TYPE MATCHES Release)
    find_package(OpenMP)
endif()

add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)
find_package(Boost REQUIRED program_options)

add_executable(minecrack-slime slime/main.cpp slime/slime_logic.cpp)
target_link_libraries(minecrack-slime Boost::program_options)

add_executable(minecrack-biome biome/main.cpp biome/biome_logic.cpp)
target_include_directories(minecrack-biome PRIVATE cubiomes)
target_link_libraries(minecrack-biome Boost::program_options cubiomes)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minecrack-slime OpenMP::OpenMP_CXX)
    target_link_libraries(minecrack-biome OpenMP::OpenMP_CXX)
endif()
