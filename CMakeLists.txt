project(app)

cmake_minimum_required(VERSION 3.0)

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Please install CUDA")
endif()

find_package(OpenCV 3 REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "Please install OpenCV 3")
endif()

add_executable(app app.cc)
target_include_directories(app PUBLIC ${CUDA_TOOLKIT_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(app nvinfer ${CUDA_CUDART_LIBRARY} ${OpenCV_LIBS})
