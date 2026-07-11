cmake_minimum_required(VERSION 3.10)
project(FaceDetectionApp)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 若未指定构建类型，默认使用 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 默认导出 compile_commands.json（便于 IDE/clangd 索引）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SFACE_MODEL "${CMAKE_SOURCE_DIR}/models/face_recognition_sface_2021dec.onnx")
set(SFACE_MODEL_SHA256 "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79")
if(EXISTS "${SFACE_MODEL}")
    file(SHA256 "${SFACE_MODEL}" SFACE_MODEL_ACTUAL_SHA256)
    if(NOT SFACE_MODEL_ACTUAL_SHA256 STREQUAL SFACE_MODEL_SHA256)
        message(FATAL_ERROR "Bundled SFace model checksum mismatch")
    endif()
else()
    message(FATAL_ERROR "Missing bundled SFace model: ${SFACE_MODEL}")
endif()

# ============================================================================
# 0. SIMD 加速选项（作用于 libfacedetection 动态库）
#    X86/X64 CPU:  cmake -DENABLE_AVX2=ON 或 -DENABLE_AVX512=ON
#    ARM CPU:      cmake -DENABLE_NEON=ON
#    开启后会自动定义对应宏并添加编译器指令集标志，无需手动改头文件。
#    AVX512 / AVX2 / NEON 三者互斥（与头文件中的 #error 检查一致）。
# ============================================================================
set(ENABLE_AVX2   ON  CACHE BOOL "Enable AVX2 acceleration (X64 CPU)")
set(ENABLE_AVX512 OFF CACHE BOOL "Enable AVX512 acceleration (X64 CPU)")
set(ENABLE_NEON   OFF CACHE BOOL "Enable NEON acceleration (ARM CPU)")

# 三者最多只能开启其中一个
set(_SIMD_ENABLED_COUNT 0)
foreach(_simd_opt ENABLE_AVX512 ENABLE_AVX2 ENABLE_NEON)
    if(${_simd_opt})
        math(EXPR _SIMD_ENABLED_COUNT "${_SIMD_ENABLED_COUNT} + 1")
    endif()
endforeach()
if(_SIMD_ENABLED_COUNT GREATER 1)
    message(FATAL_ERROR "ENABLE_AVX512 / ENABLE_AVX2 / ENABLE_NEON 三者互斥，最多只能开启其中一个。")
endif()

# 检测编译器是否支持 -mfpu=neon：
#   - 32 位 ARM (armhf) 需要该标志才能使用 arm_neon.h 内联函数；
#   - AArch64 下 NEON 默认开启，x86 不识别该标志，这两种情况检测结果均为否。
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mfpu=neon" CXX_HAS_MFPU_NEON)

# ============================================================================
# 1. 依赖查找
# ============================================================================
find_package(OpenCV REQUIRED)        # app / network_camera_receiver 需要
find_package(OpenMP)                 # libfacedetection 可选加速
find_package(Threads REQUIRED)       # network_camera_receiver 后台线程 / app

message(STATUS "Found OpenCV Version: ${OpenCV_VERSION}")
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP CXX: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP not found, building without OpenMP support.")
endif()
message(STATUS "Threads library enabled.")

# ============================================================================
# 2. 子模块
#    libfacedetection       -> SHARED 动态库（人脸检测）
#    network_camera_receiver-> STATIC 静态库（仅抓帧）
#    app                    -> 可执行程序（编排 收帧->检测->显示）
# ============================================================================
add_subdirectory(libfacedetection)
add_subdirectory(network_camera_receiver)
add_subdirectory(face_pipeline)
add_subdirectory(app)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
