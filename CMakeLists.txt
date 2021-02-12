cmake_minimum_required(VERSION 2.8)
project(m-queens2)

option(WITH_BOINC "Build only stuff needed for BOINC" OFF)

set(BUILD_STATIC_TXT  "Statically link against against libraries")
if (WIN32)
    option(BUILD_STATIC ${BUILD_STATIC_TXT} ON)
else()
    option(BUILD_STATIC ${BUILD_STATIC_TXT} OFF)
endif()

option(WITH_STANDALONE "Standalone m-queens2 binary" ON)
# for static linking a path to the boinc static build is needed
# boinc build commands:
# mkdir -p /build/static_boinc/
# ./configure --disable-server --disable-manager --disable-fcgi --disable-client --prefix=/build/static_boinc/
# make
# make install
set(BOINC_STATIC_DIR "C:/build/static_boinc/" CACHE PATH "BOINC static build directory")

set (CMAKE_CXX_STANDARD 11)

# Put OpenCL kernel into binary
set(CL_KERNEL_SRC "clqueens.cl")
set(CL_KERNEL_HEADER "clqueens_kernel.h")

add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${CL_KERNEL_HEADER}"
        COMMAND ${CMAKE_COMMAND} ARGS
        -DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_SRC}"
        -DBIN2H_VARIABLE_NAME=ethash_cl
        -DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/${CL_KERNEL_HEADER}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
        COMMENT "Generating OpenCL Kernel Byte Array"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_SRC}
)
add_custom_target(clbin2h_stable DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CL_KERNEL_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_SRC})

if (${WITH_STANDALONE})

find_package(OpenCL REQUIRED)

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
add_dependencies(${PROJECT_NAME} clbin2h_stable)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)

endif()

if(${WITH_BOINC})

set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

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
#if (CMAKE_BUILD_TYPE STREQUAL Release)
#    add_custom_command(TARGET ${PRESOLVER_PROJECT} POST_BUILD
#            COMMAND ${CMAKE_STRIP} ${PRESOLVER_PROJECT})
#endif ()
target_include_directories(${PRESOLVER_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(${PRESOLVER_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (${BUILD_STATIC})
set(CMAKE_EXE_LINKER_FLAGS " -static")
target_link_libraries(${PRESOLVER_PROJECT} "-static-libgcc -static-libstdc++")
endif()

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
add_dependencies(${BOINC_PROJECT} clbin2h_stable)

# Strip binary for release builds
#if (CMAKE_BUILD_TYPE STREQUAL Release)
#    add_custom_command(TARGET ${BOINC_PROJECT} POST_BUILD
#            COMMAND ${CMAKE_STRIP} ${BOINC_PROJECT})
#endif ()
target_include_directories(${BOINC_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(${BOINC_PROJECT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${BOINC_PROJECT} PRIVATE ${BOINC_STATIC_DIR}/include/)

if (${BUILD_STATIC})
    # Linking standard libs statically
    set(CMAKE_CXX_STANDARD_LIBRARIES "-Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "--static --verbose")
    target_link_libraries(${BOINC_PROJECT} ${BOINC_STATIC_DIR}/lib/libboinc_api.a)
    target_link_libraries(${BOINC_PROJECT} ${BOINC_STATIC_DIR}/lib/libboinc.a)
else()
    target_link_libraries(${BOINC_PROJECT} boinc)
    target_link_libraries(${BOINC_PROJECT} boinc_api)
endif()

set(BOINC_PROJECT_OCL m-queens2-boinc-ocl)

find_package(OpenCL REQUIRED)

# needed for the OpenCL C++ wrapper
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# copy OpenCL kernel source
configure_file(${CMAKE_SOURCE_DIR}/clqueens.cl ./ COPYONLY)

set(${BOINC_PROJECT_OCL}_SOURCES
    boinc/main.cpp
    clqueens.cl
    clsolver.cpp
    clsolver.h
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

add_executable(m-queens2-boinc-ocl ${${BOINC_PROJECT_OCL}_SOURCES})
target_compile_definitions(m-queens2-boinc-ocl PRIVATE BOINC_OCL_SOLVER=1)

# Strip binary for release builds
if (CMAKE_BUILD_TYPE STREQUAL Release)
    set_target_properties(m-queens2-boinc-ocl PROPERTIES LINK_FLAGS_RELEASE -s)
endif ()
target_include_directories(m-queens2-boinc-ocl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(m-queens2-boinc-ocl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(m-queens2-boinc-ocl PRIVATE ${BOINC_STATIC_DIR}/include/)

target_link_libraries(m-queens2-boinc-ocl OpenCL::OpenCL)
add_dependencies(m-queens2-boinc-ocl clbin2h_stable)

if (${BUILD_STATIC})
    # Linking standard libs statically
    set(CMAKE_CXX_STANDARD_LIBRARIES "-Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "--static --verbose")
    target_link_libraries(m-queens2-boinc-ocl ${BOINC_STATIC_DIR}/lib/libboinc_api.a)
    target_link_libraries(m-queens2-boinc-ocl ${BOINC_STATIC_DIR}/lib/libboinc_opencl.a)
    target_link_libraries(m-queens2-boinc-ocl ${BOINC_STATIC_DIR}/lib/libboinc.a)
else()
    target_link_libraries(m-queens2-boinc-ocl boinc)
    target_link_libraries(m-queens2-boinc-ocl boinc_api)
endif()

endif()
