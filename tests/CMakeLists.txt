add_executable(face_pipeline_tests
    face_pipeline_tests.cpp
)

target_link_libraries(face_pipeline_tests PRIVATE
    face_pipeline
)

target_compile_definitions(face_pipeline_tests PRIVATE
    SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

add_test(NAME face_pipeline_tests COMMAND face_pipeline_tests)
