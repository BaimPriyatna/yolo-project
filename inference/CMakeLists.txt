
cmake_minimum_required(VERSION 3.16)

project(YoloCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Allow old third-party CMake projects (e.g. Eigen/ByteTrack)
# to configure with modern CMake.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ============================================================
# Dependencies
# ============================================================

# OpenCV:
# Only request modules actually used by this project.
# Do NOT use ${OpenCV_LIBS}, because some Arch OpenCV builds
# may include optional modules such as opencv_viz/VTK.
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    videoio
    highgui
    dnn
)

find_package(Eigen3 REQUIRED)

# ============================================================
# ByteTrack
# ============================================================

include(FetchContent)

set(BUILD_BYTETRACK_TEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    bytetrack
    GIT_REPOSITORY https://github.com/Vertical-Beach/ByteTrack-cpp.git
    GIT_TAG main
)

FetchContent_MakeAvailable(bytetrack)

# ============================================================
# ONNX Runtime
# ============================================================

set(ONNXRUNTIME_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-linux-x64-1.29.0
)

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers not found:\n"
        "  ${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h\n"
        "Expected ONNX Runtime 1.29.0 Linux x64."
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(FATAL_ERROR
        "ONNX Runtime library not found:\n"
        "  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    )
endif()

# ============================================================
# Executable
# ============================================================

add_executable(yolo
    src/main.cpp
    src/detector.cpp
    src/recognizer.cpp
    src/config.cpp
)

# ============================================================
# Include directories
# ============================================================

target_include_directories(yolo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(yolo PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

# ============================================================
# Link libraries
# ============================================================

# IMPORTANT:
# Explicitly link only the OpenCV modules needed by the project.
# This avoids pulling opencv_viz -> VTK dependencies.

target_link_libraries(yolo PRIVATE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_highgui
    opencv_dnn

    onnxruntime

    bytetrack

    Eigen3::Eigen
)

# ============================================================
# Runtime library search path
# ============================================================

# Allows ./yolo to find libonnxruntime.so without requiring
# LD_LIBRARY_PATH every time.

set_target_properties(yolo PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
