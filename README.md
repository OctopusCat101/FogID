# FogID

FogID 是一款基于 Rust 开发的隐形水印工具，支持在图片、视频和音频文件中嵌入和提取不可见的文本水印。采用 DWT + DCT + SVD 等信号处理算法，水印在视觉/听觉上不可感知，且能抵抗有损压缩、缩放、裁剪等常见操作。

## 功能特性

- 图片水印：DWT + DCT + SVD + QIM 算法，支持 PNG / JPEG / BMP 格式
- 视频水印：基于亮度通道的块均值 QIM，多帧累积提取，支持 MP4 / MKV / AVI / MOV / WebM / FLV
- 音频水印：FFT 频谱带调制，抗有损编码，支持 MP3 / WAV / FLAC / AAC / M4A / OGG
- 密码保护：可选的 SHA-256 密钥派生 + 加密混洗
- 现代 GUI：基于 egui 的深色主题界面，支持中文显示

## 技术架构

```
src/
├── main.rs                 # 程序入口，字体加载
├── app.rs                  # GUI 应用逻辑
├── theme.rs                # 深色主题样式
└── watermark/
    ├── mod.rs              # 模块导出
    ├── core.rs             # 核心水印算法 (DWT+DCT+SVD+QIM)
    ├── dct.rs              # 离散余弦变换
    ├── dwt.rs              # 离散小波变换 (Haar)
    ├── text.rs             # 文本编码/解码 (CRC-8 校验)
    ├── video.rs            # 视频水印 (FFmpeg 管道)
    ├── audio.rs            # 音频水印 (FFmpeg 管道)
    └── fft.rs              # FFT 频谱带引擎
```

## 环境要求

- Rust 2024 edition（需要较新版本的 Rust 工具链）
- FFmpeg：视频和音频功能依赖 FFmpeg，需安装并加入系统 PATH
- Windows：中文字体自动加载（SimHei / Microsoft YaHei）

## 构建与运行

```bash
# 构建 release 版本
cargo build --release

# 直接运行
cargo run --release
```

Release 构建已配置最大优化（opt-level 3 + LTO + 单 codegen unit）。

## 使用方式

启动后进入 GUI 界面，操作流程：

1. 选择模式：嵌入（Embed）或提取（Extract）
2. 选择媒体类型：图片 / 视频 / 音频
3. 选择输入文件
4. 嵌入模式下输入水印文本（最长 60 字节 UTF-8）
5. 可选：启用密码保护
6. 选择输出路径，点击执行

界面提供实时进度条、图片预览和状态反馈。

## 算法细节

### 图片水印

RGB -> YUV 色彩空间转换 -> 多级 Haar DWT -> LL 子带分块 DCT -> 系数加密混洗 -> SVD 分解 -> QIM 嵌入 -> 逆变换还原。U/V 通道降低调制强度以减少色彩偏移。提取时支持暴力网格对齐搜索，应对裁剪场景。

### 视频水印

FFmpeg 解码为原始 RGB 帧 -> 8x8 块亮度均值计算 -> 基于块方差的自适应 QIM -> 等量 RGB 偏移（仅调制亮度）-> FFmpeg 编码（H.264 / VP9）。提取采用多帧置信度加权累积，CRC 校验通过即提前退出。

### 音频水印

FFmpeg 解码为 32-bit float PCM -> 单声道降混用于水印计算 -> 1024 样本非重叠帧 -> FFT 频谱带配对调制 -> IFFT 增量叠加至所有声道 -> FFmpeg 编码。提取时执行样本偏移 x 帧偏移的对齐搜索网格。

### 文本编码格式

```
[Magic 2B] + [Length 1B] + [UTF-8 Text] + [CRC8 1B] + [Padding]
```

固定 64 字节（512 位），最大有效载荷 60 字节，CRC-8 校验确保数据完整性。

## 抗攻击能力

| 攻击类型 | 图片 | 视频 | 音频 |
|---------|------|------|------|
| 有损压缩 (JPEG/H.264/AAC) | 支持 | 支持 | 支持 |
| 缩放 | 支持 | - | - |
| 裁剪 | 支持（对齐搜索） | - | 支持（对齐搜索） |
| 格式转换 | 支持 | 支持 | 支持 |

## 依赖项

| 依赖 | 用途 |
|------|------|
| eframe 0.31 | GUI 框架 (egui) |
| image 0.25 | 图片读写 |
| nalgebra 0.33 | SVD 线性代数运算 |
| rand / rand_chacha | 加密随机数混洗 |
| sha2 0.10 | 密码哈希 |
| rayon 1.10 | 并行计算 |
| rustfft 6 | 音频 FFT |
| rfd 0.15 | 原生文件对话框 |

## 适用场景

- 版权保护：在媒体文件中嵌入所有权信息
- 内容溯源：追踪泄露内容的来源
- 真实性验证：校验媒体文件是否被篡改
- 内容分发监控：跟踪内容传播路径

## 已知限制

- 视频和音频功能依赖外部 FFmpeg
- 水印文本上限 60 字节
- 字体加载针对 Windows 优化
- 视频/音频处理为离线模式，不支持流式处理

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
