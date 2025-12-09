cmake_minimum_required(VERSION 3.11)
project(TensorRT_YoloV5 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(pybind11 REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

add_library(yolo_plugin SHARED
    plugins/yoloPlugins.cpp
    plugins/yoloForward_nc.cu
)
target_include_directories(yolo_plugin PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(yolo_plugin PUBLIC ${TensorRT_LIBRARIES})
set_target_properties(yolo_plugin PROPERTIES CUDA_ARCHITECTURES "87")

add_library(yolo_utils SHARED
    utils/postprocess.cpp
    utils/preprocess.cu
)
target_link_libraries(yolo_utils PUBLIC ${OpenCV_LIBS})
set_target_properties(yolo_utils PROPERTIES CUDA_ARCHITECTURES "87")

add_executable(build
    build.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build PRIVATE -Wl,--no-as-needed yolo_plugin) # -Wl,--no-as-needed is needed to avoid linking errors
target_link_libraries(build PRIVATE yolo_utils)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "87")

pybind11_add_module(yolov5_trt yolov5_infer.cu ${TensorRT_SAMPLE_DIR}/common/logger.cpp)
target_include_directories(yolov5_trt PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(yolov5_trt PRIVATE yolo_plugin yolo_utils)
set_target_properties(yolov5_trt PROPERTIES CUDA_ARCHITECTURES "87")