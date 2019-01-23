cmake_minimum_required(VERSION 2.8)

project(m-queens)
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)

# needed for the OpenCL C++ wrapper
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -mtune=native")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")

set(${PROJECT_NAME}_SOURCES
  clqueens.cl
  clsolver.cpp
  clsolver.h
  cpusolver.h
  cpusolver.cpp
  isolver.h
  presolver.cpp
  presolver.h
  serialize_util.cpp
  serialize_util.h
  solverstructs.h
  start_file_handler.cpp
  start_file_handler.h
 )

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} "main.cpp")
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)

# copy OpenCL kernel source
configure_file(${CMAKE_SOURCE_DIR}/clqueens.cl ./ COPYONLY)
