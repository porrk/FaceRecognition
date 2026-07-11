# Repository Guidelines

## Project Structure & Module Organization

This is a C++11/CMake face-analysis application. `app/` owns UI orchestration, `network_camera_receiver/` captures HTTP MJPEG frames, and `face_pipeline/` contains motion gating, tracking, alignment, embedding, and publishing. `libfacedetection/` is the shared CNN detector. Store fixtures in `resources/`, tests in `tests/`, and pinned inference assets in `models/`. Treat `build/` as generated output.

## Build, Test, and Development Commands

Install CMake 3.10+, a C++11 compiler, and OpenCV development headers. Then use:

```bash
cmake -S . -B build -DENABLE_AVX2=ON
cmake --build build -j
./build/app/face_detection_app
```

The application expects an MJPEG stream at `http://127.0.0.1:5000/video`; press Esc to exit. SIMD options are mutually exclusive. For AVX512 or ARM builds, disable AVX2 and enable the target option, for example `-DENABLE_AVX2=OFF -DENABLE_NEON=ON`. OpenMP is detected automatically.

Run `ctest --test-dir build --output-on-failure` after building. For live pipeline changes, also confirm frame reception, stable track IDs, first-event-only feature logs, and clean shutdown against an MJPEG stream.

## Coding Style & Naming Conventions

Follow the existing C++ style: four-space indentation, opening braces on the same line, and focused comments for non-obvious buffer layouts or concurrency. Use `PascalCase` for classes (`NetworkCameraReceiver`), `camelCase` for functions (`getLatestFrame`), and `snake_case` for local variables (`window_name`). Keep public declarations in `include/` and implementations in `src/`. Preserve const-correctness and use RAII/standard synchronization primitives for new resource or thread ownership. There is no configured formatter or linter, so match adjacent code closely.

## Testing Guidelines

Integrate coverage with CTest and name files after behavior, such as `network_camera_receiver_test.cpp`. Avoid tests that require a physical camera; prefer fixtures, synthetic frames, fake clocks, or a controllable local stream.

## Commit & Pull Request Guidelines

Recent history uses short, lowercase, imperative summaries such as `add readme`. Keep commits focused and describe the user-visible change in the subject. Pull requests should explain scope, build/test commands run, platform and SIMD option used, and any stream assumptions. Link related issues and include screenshots for changes to rendered detection output.
