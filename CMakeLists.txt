cmake_minimum_required(VERSION 2.8)
project(m-queens2)

option(WITH_BOINC "Build only stuff needed for BOINC" OFF)

set(BUILD_STATIC_TXT  "Statically link against against libraries")
if (WIN32)
    option(BUILD_STATIC ${BUILD_STATIC_TXT} ON)
	message("Static build")
else()
    option(BUILD_STATIC ${BUILD_STATIC_TXT} OFF)
	message("Dynamic build")
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
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_HEADER}"
        COMMAND ${CMAKE_COMMAND} ARGS
        -DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_SRC}"
        -DBIN2H_VARIABLE_NAME="CLQUEENS_CL"
        -DBIN2H_HEADER_FILE="${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_HEADER}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
        COMMENT "Generating OpenCL Kernel Byte Array"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_SRC}
)
add_custom_target(cl_kernel_lib DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/${CL_KERNEL_SRC})

# Common files used by all build targets
add_library(m-queens2-common
    isolver.h
    result_file.cpp
    result_file.h
    presolver.cpp
    presolver.h
    serialize_util.cpp
    serialize_util.h
    solverstructs.h
    start_file.cpp
    start_file.h
)

find_package(OpenCL REQUIRED)
# needed for the OpenCL C++ wrapper
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# CPU solver core
add_library(m-queens2-cpu-solver
    cpusolver.h
    cpusolver.cpp
)

# OpenCL solver core
add_library(m-queens2-ocl-solver
    clsolver.h
    clsolver.cpp
)

# OpenCL solver needs thre pre-solver from common
target_link_libraries(m-queens2-ocl-solver OpenCL::OpenCL m-queens2-common)
add_dependencies(m-queens2-ocl-solver cl_kernel_lib)

if (${WITH_STANDALONE})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -mtune=native")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")

add_executable(m-queens2 main.cpp)
target_link_libraries(m-queens2 m-queens2-common m-queens2-cpu-solver m-queens2-ocl-solver)
target_include_directories(m-queens2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
if (${BUILD_STATIC})
set(CMAKE_EXE_LINKER_FLAGS " -static")
target_link_libraries(m-queens2 "-static-libgcc -static-libstdc++")
endif()

endif()

if(${WITH_BOINC})

set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

add_executable(m-queens2-presolver "presolver/main.cpp")
target_link_libraries(m-queens2-presolver m-queens2-common m-queens2-cpu-solver)
# Strip binary for release builds
#if (CMAKE_BUILD_TYPE STREQUAL Release)
#    add_custom_command(TARGET ${PRESOLVER_PROJECT} POST_BUILD
#            COMMAND ${CMAKE_STRIP} ${PRESOLVER_PROJECT})
#endif ()
target_include_directories(m-queens2-presolver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(m-queens2-presolver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (${BUILD_STATIC})
set(CMAKE_EXE_LINKER_FLAGS " -static")
target_link_libraries(m-queens2-presolver "-static-libgcc -static-libstdc++")
endif()

# BOINC executable with CPU solver
add_executable(m-queens2-boinc-cpu "boinc/main.cpp")
target_link_libraries(m-queens2-boinc-cpu m-queens2-common m-queens2-cpu-solver)

# Strip binary for release builds
#if (CMAKE_BUILD_TYPE STREQUAL Release)
#    add_custom_command(TARGET ${BOINC_PROJECT} POST_BUILD
#            COMMAND ${CMAKE_STRIP} ${BOINC_PROJECT})
#endif ()
target_include_directories(m-queens2-boinc-cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(m-queens2-boinc-cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(m-queens2-boinc-cpu PRIVATE ${BOINC_STATIC_DIR}/include/)

if (${BUILD_STATIC})
    # Linking standard libs statically
    set(CMAKE_CXX_STANDARD_LIBRARIES "-Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "--static --verbose")
    # Static BOINC builds expose all sublibraries separately
    target_link_libraries(m-queens2-boinc-cpu ${BOINC_STATIC_DIR}/lib/libboinc_api.a)
    target_link_libraries(m-queens2-boinc-cpu ${BOINC_STATIC_DIR}/lib/libboinc.a)
else()
    # For dynamic BOINC builds there's libboinc.so to link dynamically
    target_link_libraries(m-queens2-boinc-cpu boinc)
    # libboinc_api.a is a static library, but is also needed.
    target_link_libraries(m-queens2-boinc-cpu boinc_api)
endif()


# BOINC executable with OpenCL enabled solver
add_executable(m-queens2-boinc-ocl boinc/main.cpp)
target_link_libraries(m-queens2-boinc-ocl m-queens2-common m-queens2-cpu-solver m-queens2-ocl-solver)
target_compile_definitions(m-queens2-boinc-ocl PRIVATE BOINC_OCL_SOLVER=1)

# Strip binary for release builds
if (CMAKE_BUILD_TYPE STREQUAL Release)
    set_target_properties(m-queens2-boinc-ocl PROPERTIES LINK_FLAGS_RELEASE -s)
endif ()
target_include_directories(m-queens2-boinc-ocl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts/include)
target_include_directories(m-queens2-boinc-ocl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(m-queens2-boinc-ocl PRIVATE ${BOINC_STATIC_DIR}/include/)

if (${BUILD_STATIC})
    # Linking standard libs statically
    set(CMAKE_CXX_STANDARD_LIBRARIES "-Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "--static --verbose")
    # Static BOINC builds expose all sublibraries separately
    target_link_libraries(m-queens2-boinc-ocl ${BOINC_STATIC_DIR}/lib/libboinc_api.a)
    target_link_libraries(m-queens2-boinc-ocl ${BOINC_STATIC_DIR}/lib/libboinc_opencl.a)
    target_link_libraries(m-queens2-boinc-ocl ${BOINC_STATIC_DIR}/lib/libboinc.a)
else()
    # For dynamic BOINC builds there's only libboinc.so to link to
    target_link_libraries(m-queens2-boinc-ocl boinc)
    # These are static libraries, but is also needed.
    target_link_libraries(m-queens2-boinc-ocl boinc_api)
    target_link_libraries(m-queens2-boinc-ocl boinc_opencl)
endif()

endif()
