cmake_minimum_required(VERSION 2.8)

project(m-queens)

find_package(OpenCL REQUIRED)

# needed for the OpenCL C++ wrapper
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


set(${PROJECT_NAME}_SOURCES
  clqueens.cl
  clsolver.cpp
  clsolver.h
  cpusolver.h
  cpusolver.cpp
  solverstructs.h
 )

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} "main.cpp")
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

message("$<TARGET_FILE_DIR:${PROJECT_NAME}>")

# copy OpenCL kernel source
configure_file(${CMAKE_SOURCE_DIR}/clqueens.cl ./ COPYONLY)
