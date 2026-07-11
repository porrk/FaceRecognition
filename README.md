# FaceRecognition — 实时网络视频流人脸检测

基于 CNN 的人脸检测与特征提取应用：从 HTTP 网络视频流实时抓帧，经运动粗筛后检测人脸与关键点，完成对齐、SFace 特征提取和 IoU 跟踪。

- 作者：Liu zhenyu
- 语言标准：C++11
- 构建系统：CMake ≥ 3.10
- 人脸检测核心：[libfacedetection](https://github.com/ShiqiYu/libfacedetection)（BSD 3-Clause，原作者 Shiqi Yu）

---

## 目录结构

```
FaceRecognition/
├── CMakeLists.txt                  # 顶层 CMake：SIMD 选项 + 依赖查找 + 子模块
├── app/                            # 主可执行程序
│   ├── CMakeLists.txt
│   └── src/main.cpp                # 编排：收帧 -> 检测 -> 显示
├── libfacedetection/               # 人脸检测动态库 (SHARED)
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── facedetectcnn.h
│   │   └── facedetection_export.h
│   └── src/                        # CNN 模型与推理实现
├── network_camera_receiver/        # 网络抓帧静态库 (STATIC)
│   ├── CMakeLists.txt
│   ├── include/network_camera_receiver.h
│   └── src/network_camera_receiver.cpp
├── face_pipeline/                  # 运动检测、跟踪、对齐、特征与消息接口
├── models/                         # 固定版本 SFace ONNX、许可证与校验信息
├── tests/                          # CTest 离线单元/模型测试
└── resources/                      # 测试资源
```

### 主要模块

| 模块 | 类型 | 职责 |
|------|------|------|
| `libfacedetection` | SHARED 动态库 | CNN 人脸检测（含 5 个关键点），仅导出 `facedetect_cnn` |
| `network_camera_receiver` | STATIC 静态库 | 后台线程从 HTTP 视频流持续抓取最新帧，不包含检测逻辑 |
| `face_pipeline` | STATIC 静态库 | 运动粗筛、检测、IoU 跟踪、对齐、SFace 特征和结果接口 |
| `app` | 可执行程序 `face_detection_app` | 拉帧、提交任务并显示最新跟踪结果 |

---

## 功能特性

- **低延迟流水线**：采集与检测分线程，容量为 1 的 latest-wins mailbox 不积压旧帧。
- **运动粗筛**：有运动时触发检测；有人脸时每秒保活，无 track 时每 5 秒兜底扫描。
- **CNN 人脸检测**：输出人脸框 + 置信度 + 5 个面部关键点（双眼、鼻尖、嘴角）。
- **特征提取**：五点相似变换对齐后，由 OpenCV SFace 输出 L2 归一化的 128 维向量。
- **简单跟踪**：IoU 分配 `track_id`，每个 track 仅首次通过 `ResultSink` 发布。
- **SIMD 加速**：支持 AVX2 / AVX512 / NEON，CMake 选项一键开关并自动配置编译标志。
- **OpenMP 加速**：卷积运算可选多线程并行。
- **置信度过滤**：仅显示置信度 > 60 的人脸，降低误检。

---

## 依赖

- **OpenCV**（必需，视频采集与图像显示）
- **CMake ≥ 3.10**
- **C++11** 兼容编译器（GCC / Clang / MSVC）
- **OpenMP**（可选，加速检测）
- **Threads**（必需，后台抓帧线程）

Ubuntu / Debian 安装示例：

```bash
sudo apt-get install build-essential cmake libopencv-dev
# 可选 OpenMP 通常随 gcc 自带
```

---

## 构建

### 1. 默认构建（启用 AVX2）

```bash
cd FaceRecognition
cmake -S . -B build -DENABLE_AVX2=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

生成的可执行文件：`build/app/face_detection_app`。配置阶段会校验 `models/face_recognition_sface_2021dec.onnx` 的 SHA-256。

### 2. SIMD 加速选项

`ENABLE_AVX2` / `ENABLE_AVX512` / `ENABLE_NEON` 三者**互斥**，最多开启一个。

| 选项 | 适用平台 | 默认 |
|------|----------|------|
| `ENABLE_AVX2` | X86/X64 CPU | ON |
| `ENABLE_AVX512` | 较新 X64 CPU | OFF |
| `ENABLE_NEON` | ARM CPU | OFF |

```bash
# AVX512（切换前先把默认的 AVX2 关掉）
cmake -S . -B build -DENABLE_AVX2=OFF -DENABLE_AVX512=ON

# ARM NEON
cmake -S . -B build -DENABLE_AVX2=OFF -DENABLE_NEON=ON
```

> ⚠️ 开启 CPU 不支持的指令集（如在不支持 AVX512 的机器上开 `ENABLE_AVX512`）能编译通过，但运行时会 `SIGILL` 崩溃。开启前先查询：
> ```bash
> grep -oE 'avx512[a-z]*|avx2|fma' /proc/cpuinfo | sort -u
> ```

### 3. 运行时依赖

`libfacedetection` 是动态库，运行时需能找到 `libfacedetection.so`。若未安装到系统路径，可：

```bash
export LD_LIBRARY_PATH=/home/hongshaorou/lzy_demo/FaceRecognition/build/libfacedetection:$LD_LIBRARY_PATH
```

---

## 使用方法

### 1. 准备视频流

程序默认连接 `http://127.0.0.1:5000/video`（HTTP MJPEG 流）。可使用任意能产出该接口的工具，例如 Python Flask + 摄像头：

```python
# send.py —— 简易 MJPEG 推流端（示例）
from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # 本机摄像头

def gen():
    while True:
        ok, frame = cap.read()
        if not ok: continue
        _, jpg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)
```

推流端也可换成树莓派、IP 摄像头等任何能提供 `/video` MJPEG 接口的设备。

### 2. 修改目标 IP（可选）

如需连接远端主机，编辑 `app/src/main.cpp`：

```cpp
std::string host_ip = "127.0.0.1";  // 改为推流主机 IP
```

### 3. 运行

```bash
./build/app/face_detection_app
```

也可把兼容的 SFace 模型路径作为第一个参数传入。窗口显示人脸框、关键点和 `track_id`，按 **ESC** 退出。新 track 首次提取成功时输出一行 `face_feature` 日志，只记录元数据和特征维度，不打印完整向量。

---

## 检测结果解析

`facedetect_cnn` 返回的缓冲区布局（关键，写错会导致"只能检测一张脸"）：

- `buffer[0..3]`：人脸数量（`int`）
- `buffer[4..]` 起：每张人脸占 `FACEDETECTION_RESULT_STRIDE_SHORTS = 16` 个 `short`（32 字节）

每个人脸 16 个 short 字段：

| 索引 | 含义 |
|------|------|
| `p[0]` | 置信度（score × 100） |
| `p[1..4]` | 人脸框 x, y, w, h |
| `p[5..14]` | 5 个关键点（x, y 交替） |
| `p[15]` | 对齐填充 |

正确遍历写法（见 `app/src/main.cpp`）：

```cpp
short* p = ((short*)(pResults + 1)) + FACEDETECTION_RESULT_STRIDE_SHORTS * i;
```

> ⚠️ 经典版 libfacedetection（uint8 模型）每个结果占 142 字节，网上示例常写 `+ 142 * i`；在本重构版（float 模型）上会导致第 1 张起全部错位读到垃圾数据。务必使用 `FACEDETECTION_RESULT_STRIDE_SHORTS`。

---

## 项目架构流程

```
HTTP MJPEG → 抓帧线程 → 主线程运动粗筛 → latest-wins mailbox
                                      ↓
                         检测 worker：人脸检测 → IoU 跟踪
                                      ↓（仅未发布 track）
                              五点对齐 → SFace → ResultSink
```

---

## 常见问题

**Q: 编译报 `target specific option mismatch`？**
A: 开了 `_ENABLE_AVX*` 宏但没加对应 `-m` 编译标志。本项目已通过 CMake 选项自动同步两者，使用 `-DENABLE_AVX2=ON` 即可，不要手动改头文件宏。

**Q: 运行时 `SIGILL` 崩溃？**
A: 开了 CPU 不支持的指令集（如 AVX512）。用 `grep avx /proc/cpuinfo` 确认后重新 cmake。

**Q: 只能检测到一张脸？**
A: 结果 buffer stride 写错，必须用 `FACEDETECTION_RESULT_STRIDE_SHORTS`，不要用旧版的 142。

**Q: 连接视频流失败？**
A: 确认推流端已启动并监听对应端口；检查 `host_ip` 与端口是否正确；确认 `/video` 路由返回 `multipart/x-mixed-replace` 流。

---

## 许可证

- 本项目业务代码（`app/`、`network_camera_receiver/`）版权归 Liu zhenyu 所有。
- `libfacedetection` 遵循 BSD 3-Clause 许可证，版权归原作者 Shiqi Yu 所有，详见 `libfacedetection/include/facedetectcnn.h` 头部。
