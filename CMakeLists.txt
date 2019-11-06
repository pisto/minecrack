cmake_minimum_required(VERSION 3.9)
project(minecrack LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(Boost_USE_STATIC_LIBS ON)
add_definitions(-DBOOST_ENABLE_ASSERT_HANDLER)
find_package(Boost REQUIRED program_options)

if(CMAKE_BUILD_TYPE MATCHES Release)
    find_package(OpenMP)
endif()

add_executable(minecrack 48bit/main.cpp 48bit/slime_logic.cpp)

target_link_libraries(minecrack Boost::program_options)
if(OpenMP_CXX_FOUND)
    target_link_libraries(minecrack OpenMP::OpenMP_CXX)
endif()
