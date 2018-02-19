cmake_minimum_required(VERSION 2.8)

project(m-queens)
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)

# needed for the OpenCL C++ wrapper
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


set(${PROJECT_NAME}_SOURCES
  clqueens.cl
  clqueens_amd.cl
  clsolver.cpp
  clsolver.h
  cpusolver.h
  cpusolver.cpp
  presolver.cpp
  presolver.h
  solverstructs.h
 )

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} "main.cpp")
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

# copy OpenCL kernel source
configure_file(${CMAKE_SOURCE_DIR}/clqueens.cl ./ COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/clqueens_amd.cl ./ COPYONLY)
