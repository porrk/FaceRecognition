# ============================================================================
# app —— 主可执行程序
# 编排流程：1. 从 network_camera_receiver 取最新帧；
#           2. 调用 libfacedetection 检测；
#           3. 绘制并显示结果。
# ============================================================================

add_executable(face_detection_app
    src/main.cpp
)

# 链接动态库 facedetection（传递其 PUBLIC include 目录）与静态库 receiver
target_link_libraries(face_detection_app PRIVATE
    facedetection
    network_camera_receiver
    ${OpenCV_LIBS}
)

# 将项目根目录作为宏传给代码，便于用绝对路径访问 resources 等资源文件
target_compile_definitions(face_detection_app PRIVATE
    SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(face_detection_app PRIVATE -O3)
elseif(MSVC)
    target_compile_options(face_detection_app PRIVATE /O2)
endif()
