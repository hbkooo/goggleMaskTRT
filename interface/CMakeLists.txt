cmake_minimum_required(VERSION 3.2)


PROJECT (goggleMaskTest)
MESSAGE (STATUS "Project: goggleMaskRT ")  # retinaFaceReImp
MESSAGE (STATUS "Project Directory: ${PROJECT_SOURCE_DIR}")

add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig)
# find protobuf3.x
pkg_search_module(PROTOBUF REQUIRED protobuf)


set(TENSORRT_ROOT $ENV{HOME}/software/TensorRT-6.0.1.5)
include_directories (
    "./"
    ${PROTOBUF_INCLUDE_DIRS}
    ${OpenCV_INLCUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

AUX_SOURCE_DIRECTORY(./ DIR_SRCS)

message(STATUS "source file : ${DIR_SRCS}")


link_directories(
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${TENSORRT_ROOT}/lib)

cuda_add_executable(goggleMask ${DIR_SRCS} ${DIR_SRCS_CUDA} ${core_cuda_files})


target_link_libraries(goggleMask
        ${OpenCV_LIBRARIES}
        /home/hbk/project/goggleMaskRt/interface/lib/libgogglemask.so
        glog boost_system
        cudnn cublas cudart
        nvcaffe_parser nvinfer
        nppig nppicc nppc nppidei nppist)

