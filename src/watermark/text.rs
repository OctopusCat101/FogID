//! Encode/decode watermark text as a compact binary bit array.
//!
//! Format: [0xFD, 0x1D] (magic) + [1 byte length] + [N bytes UTF-8] + [1 byte CRC8]
//! Each byte is expanded to 8 bits, each bit mapped to 0 or 255
//! for compatibility with the QIM embedding pipeline.

/// Magic bytes to verify successful extraction.
const MAGIC: [u8; 2] = [0xFD, 0x1D];

/// Fixed total watermark size in bytes. All watermarks are padded to this size,
/// so extraction always uses the same `wm_bit_count` regardless of text length.
/// Max text payload = FIXED_WM_BYTES - 4 (magic 2B + length 1B + CRC 1B).
pub const FIXED_WM_BYTES: usize = 64;

/// Fixed total watermark size in bits.
pub const FIXED_WM_BITS: usize = FIXED_WM_BYTES * 8;

/// CRC-8 with polynomial 0x07 (x^8 + x^2 + x + 1).
fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Encode watermark text into a bit array (each element is 0 or 255).
pub fn text_to_bits(text: &str) -> Result<Vec<u8>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("水印文字不能为空".into());
    }
    let utf8 = text.as_bytes();
    let max_payload = FIXED_WM_BYTES - 4; // magic(2) + len(1) + CRC(1)
    if utf8.len() > max_payload {
        return Err(format!("水印文字过长（最多 {} 字节 UTF-8）", max_payload));
    }

    // Header: magic (2) + length (1) + payload + CRC8 (1) + zero-padding
    let mut payload = Vec::with_capacity(FIXED_WM_BYTES);
    payload.extend_from_slice(&MAGIC);
    payload.push(utf8.len() as u8);
    payload.extend_from_slice(utf8);
    payload.push(crc8(&payload));
    // Pad to fixed size with zeros — extraction ignores bytes after CRC
    payload.resize(FIXED_WM_BYTES, 0);

    // Expand each byte into 8 bits (MSB first), mapped to 0/255
    let mut bits = Vec::with_capacity(FIXED_WM_BITS);
    for &byte in &payload {
        for i in (0..8).rev() {
            bits.push(if (byte >> i) & 1 == 1 { 255u8 } else { 0u8 });
        }
    }

    Ok(bits)
}

/// Decode extracted bit array back to text.
///
/// `raw_bits` contains values in [0, 255]; each is thresholded at 128.
pub fn bits_to_text(raw_bits: &[u8]) -> Result<String, String> {
    // Need at least header: 2 (magic) + 1 (length) + 1 (CRC) = 4 bytes = 32 bits
    if raw_bits.len() < 32 {
        return Err("提取数据不足".into());
    }

    // Collapse bits back to bytes
    let byte_count = raw_bits.len() / 8;
    let mut bytes = Vec::with_capacity(byte_count);
    for chunk in raw_bits.chunks(8) {
        if chunk.len() < 8 {
            break;
        }
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            if bit >= 128 {
                byte |= 1 << (7 - i);
            }
        }
        bytes.push(byte);
    }

    // Verify magic
    if bytes.len() < 4 || bytes[0] != MAGIC[0] || bytes[1] != MAGIC[1] {
        return Err("未检测到水印（魔数不匹配）".into());
    }

    let text_len = bytes[2] as usize;
    if bytes.len() < 4 + text_len {
        return Err("提取数据不完整".into());
    }

    // Verify CRC8
    let crc_idx = 3 + text_len;
    let expected_crc = crc8(&bytes[..crc_idx]);
    if bytes[crc_idx] != expected_crc {
        return Err("水印校验失败（CRC 不匹配）".into());
    }

    let text_bytes = &bytes[3..3 + text_len];
    String::from_utf8(text_bytes.to_vec()).map_err(|_| "水印文字解码失败（无效 UTF-8）".into())
}

/// Quick check: do the first two bytes match the magic?
/// Used for fast filtering during brute-force alignment search.
pub fn check_watermark_magic(raw_bits: &[u8]) -> bool {
    if raw_bits.len() < 16 {
        return false;
    }
    let mut byte0 = 0u8;
    let mut byte1 = 0u8;
    for i in 0..8 {
        if raw_bits[i] >= 128 {
            byte0 |= 1 << (7 - i);
        }
        if raw_bits[8 + i] >= 128 {
            byte1 |= 1 << (7 - i);
        }
    }
    byte0 == MAGIC[0] && byte1 == MAGIC[1]
}
