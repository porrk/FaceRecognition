# ============================================================================
# network_camera_receiver —— 静态库
# 仅负责从网络视频流抓取最新帧，不包含任何人脸检测逻辑、不依赖 libfacedetection。
# ============================================================================

add_library(network_camera_receiver STATIC
    src/network_camera_receiver.cpp
)

target_include_directories(network_camera_receiver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# OpenCV 用于视频采集；Threads 用于后台抓帧线程。
# 设为 PUBLIC，链接本库的目标可传递获得这些依赖。
target_link_libraries(network_camera_receiver PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(network_camera_receiver PRIVATE -O3)
elseif(MSVC)
    target_compile_options(network_camera_receiver PRIVATE /O2)
endif()
