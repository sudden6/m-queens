cmake_minimum_required(VERSION 2.8)

project(m-queens)

set(${PROJECT_NAME}_SOURCES
  clsolver.cpp
  clsolver.h
 )

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} "main.cpp")
