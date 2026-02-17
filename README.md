# FogID

**English | [中文](README_zh.md)**

FogID is an invisible watermarking tool built with Rust, supporting embedding and extracting invisible text watermarks in images, videos, and audio files. Using signal processing algorithms like DWT + DCT + SVD, watermarks are visually/audibly imperceptible and resistant to common operations such as lossy compression, scaling, and cropping.

## Features

- **Image Watermarking**: DWT + DCT + SVD + QIM algorithm, supports PNG / JPEG / BMP formats
- **Video Watermarking**: Block-mean QIM on luminance channel, multi-frame cumulative extraction, supports MP4 / MKV / AVI / MOV / WebM / FLV
- **Audio Watermarking**: FFT spectral band modulation, resistant to lossy encoding, supports MP3 / WAV / FLAC / AAC / M4A / OGG
- **Password Protection**: Optional SHA-256 key derivation + encrypted shuffling
- **Modern GUI**: Dark-themed interface based on egui with CJK font support

## Architecture

```
src/
├── main.rs                 # Entry point, font loading
├── app.rs                  # GUI application logic
├── theme.rs                # Dark theme styling
└── watermark/
    ├── mod.rs              # Module exports
    ├── core.rs             # Core watermark algorithm (DWT+DCT+SVD+QIM)
    ├── dct.rs              # Discrete Cosine Transform
    ├── dwt.rs              # Discrete Wavelet Transform (Haar)
    ├── text.rs             # Text encoding/decoding (CRC-8 checksum)
    ├── video.rs            # Video watermarking (FFmpeg pipeline)
    ├── audio.rs            # Audio watermarking (FFmpeg pipeline)
    └── fft.rs              # FFT spectral band engine
```

## Requirements

- Rust 2024 edition (requires a recent Rust toolchain)
- FFmpeg: Required for video and audio features, must be installed and available in system PATH
- Windows: Chinese fonts auto-loaded (SimHei / Microsoft YaHei)

## Build & Run

```bash
# Build release version
cargo build --release

# Run directly
cargo run --release
```

Release builds are configured with maximum optimization (opt-level 3 + LTO + single codegen unit).

## Usage

After launching, the GUI interface provides the following workflow:

1. Select mode: Embed or Extract
2. Select media type: Image / Video / Audio
3. Select input file
4. In embed mode, enter watermark text (max 60 bytes UTF-8)
5. Optional: Enable password protection
6. Select output path and execute

The interface provides real-time progress bar, image preview, and status feedback.

## Algorithm Details

### Image Watermarking

RGB -> YUV color space conversion -> Multi-level Haar DWT -> LL subband block DCT -> Coefficient encrypted shuffling -> SVD decomposition -> QIM embedding -> Inverse transform reconstruction. U/V channels use reduced modulation strength to minimize color shift. Extraction supports brute-force grid alignment search for cropping scenarios.

### Video Watermarking

FFmpeg decodes to raw RGB frames -> 8x8 block luminance mean calculation -> Variance-based adaptive QIM -> Equal RGB offset (luminance-only modulation) -> FFmpeg encoding (H.264 / VP9). Extraction uses multi-frame confidence-weighted accumulation with early exit on CRC validation.

### Audio Watermarking

FFmpeg decodes to 32-bit float PCM -> Mono downmix for watermark calculation -> 1024-sample non-overlapping frames -> FFT spectral band pair modulation -> IFFT delta overlay to all channels -> FFmpeg encoding. Extraction performs sample offset × frame offset alignment grid search.

### Text Encoding Format

```
[Magic 2B] + [Length 1B] + [UTF-8 Text] + [CRC8 1B] + [Padding]
```

Fixed 64 bytes (512 bits), maximum payload 60 bytes, CRC-8 checksum ensures data integrity.

## Attack Resistance

| Attack Type | Image | Video | Audio |
|-------------|-------|-------|-------|
| Lossy Compression (JPEG/H.264/AAC) | ✓ | ✓ | ✓ |
| Scaling | ✓ | - | - |
| Cropping | ✓ (alignment search) | - | ✓ (alignment search) |
| Format Conversion | ✓ | ✓ | ✓ |

## Dependencies

| Dependency | Purpose |
|------------|----------|
| eframe 0.31 | GUI framework (egui) |
| image 0.25 | Image I/O |
| nalgebra 0.33 | SVD linear algebra operations |
| rand / rand_chacha | Cryptographic random shuffling |
| sha2 0.10 | Password hashing |
| rayon 1.10 | Parallel computation |
| rustfft 6 | Audio FFT |
| rfd 0.15 | Native file dialogs |

## Use Cases

- **Copyright Protection**: Embed ownership information in media files
- **Content Tracing**: Track the source of leaked content
- **Authenticity Verification**: Verify if media files have been tampered with
- **Distribution Monitoring**: Track content distribution paths

## Known Limitations

- Video and audio features require external FFmpeg
- Watermark text limited to 60 bytes
- Font loading optimized for Windows
- Video/audio processing is offline mode, streaming not supported

## License

This project is open-sourced under the [MIT License](LICENSE).
