cmake_minimum_required(VERSION 3.14...3.27)

project(RocksDBExample LANGUAGES C CXX)

# Explicitly enable C support
enable_language(C)
if(NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR "C compiler not found!")
endif()

# C++17 settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE re)

# CLI11
include(FetchContent)
FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.3.2
)
FetchContent_MakeAvailable(CLI11)

# Find packages before adding RocksDB
find_package(ZLIB REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# =======================================================
# GFLAGS
# =======================================================

if(NOT GFLAGS_INCLUDE_DIR OR NOT GFLAGS_LIBRARIES)
    message(FATAL_ERROR "GFLAGS_INCLUDE_DIR and GFLAGS_LIBRARIES must be defined (e.g., via -D arguments).")
endif()

# 2. Define an INTERFACE target that uses the provided paths.
# This target is independent of the hard-coded user name.
add_library(gflags::gflags INTERFACE IMPORTED)
set_target_properties(gflags::gflags PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GFLAGS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${GFLAGS_LIBRARIES}"
)

message(STATUS "Using gflags (paths provided by user script): Include=${GFLAGS_INCLUDE_DIR}, Library=${GFLAGS_LIBRARIES}")

# =======================================================
# GFLAGS END
# =======================================================

# Add RocksDB submodule with options
option(WITH_ZLIB "Build with ZLIB support" ON)
option(WITH_SNAPPY "Build with SNAPPY support" ON)

add_subdirectory(rocksdb)

# Main targets
add_executable(green-compressed-storage green-compressed-storage.cpp)

target_link_libraries(green-compressed-storage PRIVATE
    rocksdb
    ZLIB::ZLIB
    Arrow::arrow_shared
    Parquet::parquet_shared
    CLI11::CLI11
    gflags::gflags
)

# Standard properties
set_target_properties(green-compressed-storage PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)