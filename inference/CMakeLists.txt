cmake_minimum_required(VERSION 3.16)

project(YoloCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# --- ByteTrack-cpp: didownload otomatis via FetchContent (butuh internet saat
# cmake configure pertama kali; setelah itu di-cache di build/_deps). ---
include(FetchContent)
set(BUILD_BYTETRACK_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  bytetrack
  GIT_REPOSITORY https://github.com/Vertical-Beach/ByteTrack-cpp.git
  GIT_TAG main
)
FetchContent_MakeAvailable(bytetrack)

set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-linux-x64-1.29.0)

add_executable(yolo
  src/main.cpp
  src/detector.cpp
  src/recognizer.cpp
)

target_include_directories(yolo PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(yolo PRIVATE ${ONNXRUNTIME_ROOT}/lib)

target_link_libraries(yolo PRIVATE
  ${OpenCV_LIBS}
  onnxruntime
  bytetrack
  Eigen3::Eigen
)
