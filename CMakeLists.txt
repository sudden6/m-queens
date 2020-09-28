cmake_minimum_required(VERSION 2.8)
project(m-queens2)

option(NO_BOINC "Disable build of BOINC modules" ON)
option(BOINC_ONLY "Build only stuff needed for BOINC" OFF)
option(BUILD_STATIC "Build static binaries" ON)
# for static linking a path to the boinc static build is needed
# boinc build commands:
# mkdir -p /build/static_boinc/
# ./configure --disable-server --disable-manager --disable-fcgi --disable-client --prefix=/build/static_boinc/
# make
# make install
set(BOINC_STATIC_DIR "/build/static_boinc/lib64" CACHE PATH "BOINC static build directory")

set (CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (NOT ${BOINC_ONLY} AND NOT ${BUILD_STATIC})

find_package(OpenCL REQUIRED)

# needed for the OpenCL C++ wrapper
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=x86-64")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=x86-64")

set(${PROJECT_NAME}_SOURCES
  clqueens.cl
  clsolver.cpp
  clsolver.h
  cpusolver.h
  cpusolver.cpp
  isolver.h
  presolver.cpp
  presolver.h
  result_file.cpp
  result_file.h
  serialize_util.cpp
  serialize_util.h
  solverstructs.h
  start_file.cpp
  start_file.h
 )

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} "main.cpp")
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/parallel_hashmap)

# copy OpenCL kernel source
configure_file(${CMAKE_SOURCE_DIR}/clqueens.cl ./ COPYONLY)

endif()

set(PRESOLVER_PROJECT m-queens2-presolver)

set(${PRESOLVER_PROJECT}_SOURCES
    presolver/main.cpp
    presolver.cpp
    presolver.h
    solverstructs.h
    serialize_util.cpp
    serialize_util.h
    start_file.cpp
    start_file.h
)

add_executable(${PRESOLVER_PROJECT} ${${PRESOLVER_PROJECT}_SOURCES} "presolver/main.cpp")
# Strip binary for release builds
if (CMAKE_BUILD_TYPE STREQUAL Release)
    add_custom_command(TARGET ${PRESOLVER_PROJECT} POST_BUILD
            COMMAND ${CMAKE_STRIP} ${PRESOLVER_PROJECT})
endif ()
target_include_directories(${PRESOLVER_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(${PRESOLVER_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (${BUILD_STATIC})
set(CMAKE_EXE_LINKER_FLAGS " -static")
target_link_libraries(${PRESOLVER_PROJECT} "-static-libgcc -static-libstdc++")
endif()

if (NOT ${NO_BOINC})
set(BOINC_PROJECT m-queens2-boinc)

set(${BOINC_PROJECT}_SOURCES
    boinc/main.cpp
    cpusolver.h
    cpusolver.cpp
    presolver.cpp
    presolver.h
    result_file.cpp
    result_file.h
    solverstructs.h
    serialize_util.cpp
    serialize_util.h
    start_file.cpp
    start_file.h
)

add_executable(${BOINC_PROJECT} ${${BOINC_PROJECT}_SOURCES} "boinc/main.cpp")
# Strip binary for release builds
if (CMAKE_BUILD_TYPE STREQUAL Release)
    add_custom_command(TARGET ${BOINC_PROJECT} POST_BUILD
            COMMAND ${CMAKE_STRIP} ${BOINC_PROJECT})
endif ()
target_include_directories(${BOINC_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(${BOINC_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (${BUILD_STATIC})
    # Linking standard libs statically
    set(CMAKE_CXX_STANDARD_LIBRARIES "-Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS ${OpenMP_EXE_LINKER_FLAGS} "--static --verbose")
    target_include_directories(${BOINC_PROJECT} PRIVATE ${BOINC_STATIC_DIR}/include)
    find_library(BOINC_LIB NAMES libboinc.a PATHS ${BOINC_STATIC_DIR}/lib NO_DEFAULT_PATH)
    find_library(BOINC_API NAMES libboinc_api.a PATHS ${BOINC_STATIC_DIR}/lib NO_DEFAULT_PATH)

    target_link_libraries(${BOINC_PROJECT} ${BOINC_API})
    target_link_libraries(${BOINC_PROJECT} ${BOINC_LIB})
else()
    target_link_libraries(${BOINC_PROJECT} boinc)
    target_link_libraries(${BOINC_PROJECT} boinc_api)
endif()
endif()


