add_library(face_pipeline STATIC
    src/motion_detector.cpp
    src/face_detector.cpp
    src/face_aligner.cpp
    src/face_embedder.cpp
    src/iou_tracker.cpp
    src/result_sink.cpp
    src/face_pipeline.cpp
)

target_include_directories(face_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(face_pipeline PUBLIC
    facedetection
    ${OpenCV_LIBS}
    Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(face_pipeline PRIVATE -O3)
elseif(MSVC)
    target_compile_options(face_pipeline PRIVATE /O2)
endif()
