// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Reader for `.spz` Gaussian splat files, ported from the `spz` library in
//! `3rdparty/spz` (`load-spz.cc`, `splat-types.h`).
//!
//! Two container formats exist: the current NGSP format (version 4), a 32-byte
//! header followed by a table of contents and one zstd stream per attribute, and
//! the legacy format (versions 1 to 3), a single gzip stream holding a 16-byte
//! header and the attributes. Both are read into [`PackedGaussians`], which
//! [`PackedGaussians::unpack`] decodes one Gaussian at a time, as the C++ library does.

use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

/// The highest spherical-harmonics degree the format stores.
pub const SH_MAX_DEGREE: u8 = 4;
/// Spherical-harmonics coefficients per colour channel at degree 4.
pub const SH_MAX_COEFFS: usize = 24;
/// The newest header version.
pub const LATEST_SPZ_HEADER_VERSION: u32 = 4;
/// Versions from this one on are NGSP files with zstd streams.
pub const MIN_ZSTD_SPZ_HEADER_VERSION: u32 = 4;
/// Versions from this one on store rotations as "smallest three" quaternions.
pub const MIN_SMALLEST_THREE_QUATERNIONS_VERSION: u32 = 3;
/// "NGSP" in little-endian byte order.
pub const NGSP_MAGIC: u32 = 0x5053_474e;

const NGSP_HEADER_SIZE: usize = 32;
const LEGACY_HEADER_SIZE: usize = 16;
const FLAG_ANTIALIASED: u8 = 0x1;

// Scale factor of the DC colour components (see load-spz.cc).
const COLOR_SCALE: f32 = 0.15;
const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Number of spherical-harmonics coefficients per channel for a degree.
pub fn dim_for_degree(degree: u8) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        3 => 15,
        4 => 24,
        _ => 0,
    }
}

/// Axis conventions a scene can be stored in; the names give the directions of +x, +y and +z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateSystem {
    Unspecified = 0,
    /// Left Down Back
    Ldb = 1,
    /// Right Down Back
    Rdb = 2,
    /// Left Up Back
    Lub = 3,
    /// Right Up Back, the Three.js coordinate system
    Rub = 4,
    /// Left Down Front
    Ldf = 5,
    /// Right Down Front, the PLY coordinate system
    Rdf = 6,
    /// Left Up Front, the GLB coordinate system
    Luf = 7,
    /// Right Up Front, the Unity coordinate system
    Ruf = 8,
}

/// Sign flips that convert positions, rotations and spherical harmonics between coordinate systems.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateConverter {
    /// Flips of the x, y and z position components.
    pub flip_p: [f32; 3],
    /// Flips of the quaternion's x, y and z components; w is never flipped.
    pub flip_q: [f32; 3],
    /// Flips of the 24 spherical-harmonics coefficients.
    pub flip_sh: [f32; SH_MAX_COEFFS],
}

impl Default for CoordinateConverter {
    fn default() -> Self {
        Self {
            flip_p: [1.0; 3],
            flip_q: [1.0; 3],
            flip_sh: [1.0; SH_MAX_COEFFS],
        }
    }
}

fn axes_match(a: CoordinateSystem, b: CoordinateSystem) -> [bool; 3] {
    let a_num = a as i32 - 1;
    let b_num = b as i32 - 1;
    if a_num < 0 || b_num < 0 {
        return [true, true, true];
    }
    [
        ((a_num >> 0) & 1) == ((b_num >> 0) & 1),
        ((a_num >> 1) & 1) == ((b_num >> 1) & 1),
        ((a_num >> 2) & 1) == ((b_num >> 2) & 1),
    ]
}

/// The converter from `from` to `to`, as `spz::coordinateConverter` computes it.
pub fn coordinate_converter(from: CoordinateSystem, to: CoordinateSystem) -> CoordinateConverter {
    let [x_match, y_match, z_match] = axes_match(from, to);
    let x = if x_match { 1.0 } else { -1.0 };
    let y = if y_match { 1.0 } else { -1.0 };
    let z = if z_match { 1.0 } else { -1.0 };
    CoordinateConverter {
        flip_p: [x, y, z],
        flip_q: [y * z, x * z, x * y],
        flip_sh: [
            y,
            z,
            x,
            x * y,
            y * z,
            1.0,
            x * z,
            1.0,
            y,
            x * y * z,
            y,
            z,
            x,
            z,
            x,
            x * y,
            y * z,
            x * y,
            y * z,
            1.0,
            x * z,
            1.0,
            x * z,
            y,
        ],
    }
}

/// One decoded Gaussian with the fields of `spz::UnpackedGaussian`: log scales, logit alpha, SH0 colour.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UnpackedGaussian {
    pub position: [f32; 3],
    /// x, y, z, w
    pub rotation: [f32; 4],
    /// Natural logarithm of the scale.
    pub scale: [f32; 3],
    /// DC spherical-harmonics colour.
    pub color: [f32; 3],
    /// Opacity before the sigmoid.
    pub alpha: f32,
    pub sh_r: [f32; SH_MAX_COEFFS],
    pub sh_g: [f32; SH_MAX_COEFFS],
    pub sh_b: [f32; SH_MAX_COEFFS],
}

/// The quantized attributes of all Gaussians of a file, as stored.
#[derive(Clone, Debug, Default)]
pub struct PackedGaussians {
    /// Header version of the file.
    pub version: u32,
    /// Number of Gaussians.
    pub num_points: usize,
    /// Spherical-harmonics degree, 0 to 4.
    pub sh_degree: u8,
    /// Fractional bits of the 24-bit fixed-point positions.
    pub fractional_bits: u8,
    /// Whether the scene was trained with antialiasing.
    pub antialiased: bool,
    /// Rotations are 4-byte "smallest three" quaternions (else 3 bytes, first three components).
    pub uses_quaternion_smallest_three: bool,
    /// 9 bytes per Gaussian (24-bit fixed point), or 6 bytes in the legacy float16 encoding.
    pub positions: Vec<u8>,
    pub scales: Vec<u8>,
    pub rotations: Vec<u8>,
    pub alphas: Vec<u8>,
    pub colors: Vec<u8>,
    /// `dim_for_degree(sh_degree)` coefficients of three channels per Gaussian, interleaved r, g, b.
    pub sh: Vec<u8>,
}

fn unquantize_sh(x: u8) -> f32 {
    (x as f32 - 128.0) / 128.0
}

fn inv_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

fn half_to_float(h: u16) -> f32 {
    let sign = ((h >> 15) & 0x1) as u32;
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as u32;
    let sign_multiplier = if sign == 1 { -1.0f32 } else { 1.0 };
    if exponent == 0 {
        // Subnormal numbers (no exponent, 0 in the mantissa decimal).
        return sign_multiplier * 2f32.powi(-14) * mantissa as f32 / 1024.0;
    }
    if exponent == 31 {
        // Infinity or NaN.
        return if mantissa != 0 {
            f32::NAN
        } else {
            sign_multiplier * f32::INFINITY
        };
    }
    // Non-zero exponent implies 1 in the mantissa decimal.
    sign_multiplier * 2f32.powi(exponent - 15) * (1.0 + mantissa as f32 / 1024.0)
}

fn unpack_quaternion_first_three(rotation: &mut [f32; 4], r: &[u8], c: &CoordinateConverter) {
    let mut xyz = [0.0f32; 3];
    for i in 0..3 {
        xyz[i] = (r[i] as f32 * (1.0f32 / 127.5) + -1.0) * c.flip_q[i];
    }
    rotation[..3].copy_from_slice(&xyz);
    // The quaternion is normalized and w is non-negative.
    let squared_norm = xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2];
    rotation[3] = (1.0f32 - squared_norm).max(0.0).sqrt();
}

fn unpack_quaternion_smallest_three(rotation: &mut [f32; 4], r: &[u8], c: &CoordinateConverter) {
    let mut comp = r[0] as u32 + ((r[1] as u32) << 8) + ((r[2] as u32) << 16) + ((r[3] as u32) << 24);
    const C_MASK: u32 = (1 << 9) - 1;
    let i_largest = (comp >> 30) as usize;
    let mut sum_squares = 0.0f32;
    for i in (0..4).rev() {
        if i != i_largest {
            let mag = comp & C_MASK;
            let negbit = (comp >> 9) & 0x1;
            comp >>= 10;
            rotation[i] = SQRT1_2 * mag as f32 / C_MASK as f32;
            if negbit == 1 {
                rotation[i] = -rotation[i];
            }
            // Fused, as the C++ reference build (clang on arm64) contracts it; the result is
            // then bit-identical to the C++ loader there, and at most 1 ulp off elsewhere.
            sum_squares = rotation[i].mul_add(rotation[i], sum_squares);
        }
    }
    rotation[i_largest] = (1.0f32 - sum_squares).sqrt();
    for i in 0..3 {
        rotation[i] *= c.flip_q[i];
    }
}

impl PackedGaussians {
    /// Whether positions use the legacy float16 encoding (6 bytes per Gaussian).
    pub fn uses_float16(&self) -> bool {
        self.positions.len() == self.num_points * 3 * 2
    }

    fn check_sizes(&self) -> Result<()> {
        let sh_dim = dim_for_degree(self.sh_degree);
        let n = self.num_points;
        ensure!(
            self.positions.len() == n * 3 * if self.uses_float16() { 2 } else { 3 },
            "spz: position size"
        );
        ensure!(self.scales.len() == n * 3, "spz: scale size");
        ensure!(
            self.rotations.len() == n * if self.uses_quaternion_smallest_three { 4 } else { 3 },
            "spz: rotation size"
        );
        ensure!(self.alphas.len() == n, "spz: alpha size");
        ensure!(self.colors.len() == n * 3, "spz: color size");
        ensure!(self.sh.len() == n * sh_dim * 3, "spz: sh size");
        Ok(())
    }

    /// Decodes Gaussian `i`, converting it with `c`; missing SH coefficients are zero.
    pub fn unpack(&self, i: usize, c: &CoordinateConverter) -> UnpackedGaussian {
        let mut result = UnpackedGaussian::default();
        if self.uses_float16() {
            let p = &self.positions[i * 6..i * 6 + 6];
            for k in 0..3 {
                let half = u16::from_le_bytes([p[2 * k], p[2 * k + 1]]);
                result.position[k] = c.flip_p[k] * half_to_float(half);
            }
        } else {
            let scale = (1.0f64 / (1u32 << self.fractional_bits) as f64) as f32;
            let p = &self.positions[i * 9..i * 9 + 9];
            for k in 0..3 {
                let mut fixed32 = p[k * 3] as i32;
                fixed32 |= (p[k * 3 + 1] as i32) << 8;
                fixed32 |= (p[k * 3 + 2] as i32) << 16;
                if fixed32 & 0x80_0000 != 0 {
                    fixed32 |= 0xff00_0000u32 as i32; // sign extension
                }
                result.position[k] = c.flip_p[k] * fixed32 as f32 * scale;
            }
        }

        for k in 0..3 {
            result.scale[k] = self.scales[i * 3 + k] as f32 / 16.0 - 10.0;
        }

        if self.uses_quaternion_smallest_three {
            unpack_quaternion_smallest_three(&mut result.rotation, &self.rotations[i * 4..i * 4 + 4], c);
        } else {
            unpack_quaternion_first_three(&mut result.rotation, &self.rotations[i * 3..i * 3 + 3], c);
        }

        result.alpha = inv_sigmoid(self.alphas[i] as f32 / 255.0);

        for k in 0..3 {
            result.color[k] = ((self.colors[i * 3 + k] as f32 / 255.0) - 0.5) / COLOR_SCALE;
        }

        // Coefficients the file does not store are 128, i.e. zero.
        let sh_dim = dim_for_degree(self.sh_degree);
        let sh = &self.sh[i * sh_dim * 3..(i + 1) * sh_dim * 3];
        for j in 0..SH_MAX_COEFFS {
            let (r, g, b) = if j < sh_dim {
                (sh[j * 3], sh[j * 3 + 1], sh[j * 3 + 2])
            } else {
                (128, 128, 128)
            };
            result.sh_r[j] = c.flip_sh[j] * unquantize_sh(r);
            result.sh_g[j] = c.flip_sh[j] * unquantize_sh(g);
            result.sh_b[j] = c.flip_sh[j] * unquantize_sh(b);
        }
        result
    }
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn read_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn load_ngsp(data: &[u8]) -> Result<PackedGaussians> {
    ensure!(data.len() >= NGSP_HEADER_SIZE, "spz: NGSP file too short");
    let version = read_u32(data, 4);
    let num_points = read_u32(data, 8) as usize;
    let sh_degree = data[12];
    let fractional_bits = data[13];
    let flags = data[14];
    let num_streams = data[15] as usize;
    let toc_byte_offset = read_u32(data, 16) as usize;
    ensure!(
        (MIN_ZSTD_SPZ_HEADER_VERSION..=LATEST_SPZ_HEADER_VERSION).contains(&version),
        "spz: unsupported version {version}"
    );
    ensure!(num_points > 0, "spz: invalid point count");
    ensure!(sh_degree <= SH_MAX_DEGREE, "spz: unsupported SH degree {sh_degree}");

    let uses_quaternion_smallest_three = version >= MIN_SMALLEST_THREE_QUATERNIONS_VERSION;
    let sh_dim = dim_for_degree(sh_degree);
    // The streams in the order the writer stores them; empty ones are not stored.
    let sizes = [
        num_points * 9,
        num_points,
        num_points * 3,
        num_points * 3,
        num_points * if uses_quaternion_smallest_three { 4 } else { 3 },
        num_points * sh_dim * 3,
    ];
    let present: Vec<usize> = (0..sizes.len()).filter(|&k| sizes[k] > 0).collect();
    ensure!(present.len() == num_streams, "spz: stream count mismatch");

    ensure!(toc_byte_offset >= NGSP_HEADER_SIZE, "spz: TOC overlaps the header");
    let toc_end = toc_byte_offset + num_streams * 16;
    ensure!(toc_end <= data.len(), "spz: TOC beyond the end of the file");

    let mut streams: Vec<Vec<u8>> = vec![Vec::new(); sizes.len()];
    let mut compressed_offset = toc_end;
    for (s, &k) in present.iter().enumerate() {
        let entry = toc_byte_offset + s * 16;
        let compressed_size = read_u64(data, entry) as usize;
        let uncompressed_size = read_u64(data, entry + 8) as usize;
        ensure!(uncompressed_size == sizes[k], "spz: stream size mismatch");
        let end = compressed_offset + compressed_size;
        ensure!(end <= data.len(), "spz: stream beyond the end of the file");
        let mut decoder = ruzstd::decoding::StreamingDecoder::new(&data[compressed_offset..end])
            .map_err(|e| anyhow::anyhow!("spz: zstd stream {s}: {e}"))?;
        let mut out = Vec::with_capacity(uncompressed_size);
        decoder
            .read_to_end(&mut out)
            .with_context(|| format!("spz: zstd stream {s}"))?;
        ensure!(
            out.len() == uncompressed_size,
            "spz: zstd stream {s} has the wrong size"
        );
        streams[k] = out;
        compressed_offset = end;
    }
    ensure!(compressed_offset == data.len(), "spz: compressed data size mismatch");

    let [positions, alphas, colors, scales, rotations, sh] = <[Vec<u8>; 6]>::try_from(streams).unwrap();
    Ok(PackedGaussians {
        version,
        num_points,
        sh_degree,
        fractional_bits,
        antialiased: flags & FLAG_ANTIALIASED != 0,
        uses_quaternion_smallest_three,
        positions,
        scales,
        rotations,
        alphas,
        colors,
        sh,
    })
}

fn load_legacy(compressed: &[u8]) -> Result<PackedGaussians> {
    let mut data = Vec::new();
    flate2::read::GzDecoder::new(compressed)
        .read_to_end(&mut data)
        .context("spz: gzip decompression failed")?;
    ensure!(data.len() >= LEGACY_HEADER_SIZE, "spz: header not found");
    ensure!(read_u32(&data, 0) == NGSP_MAGIC, "spz: header not found");
    let version = read_u32(&data, 4);
    let num_points = read_u32(&data, 8) as i32;
    let sh_degree = data[12];
    let fractional_bits = data[13];
    let flags = data[14];
    ensure!(
        (1..=LATEST_SPZ_HEADER_VERSION).contains(&version),
        "spz: version not supported: {version}"
    );
    ensure!(num_points > 0, "spz: invalid point count: {num_points}");
    ensure!(sh_degree <= SH_MAX_DEGREE, "spz: unsupported SH degree: {sh_degree}");
    let num_points = num_points as usize;
    let uses_float16 = version == 1;
    let uses_quaternion_smallest_three = version >= MIN_SMALLEST_THREE_QUATERNIONS_VERSION;
    let sh_dim = dim_for_degree(sh_degree);

    // Extensions may follow the attributes; they are not read.
    let mut offset = LEGACY_HEADER_SIZE;
    let mut take = |size: usize| -> Result<Vec<u8>> {
        ensure!(offset + size <= data.len(), "spz: read error");
        let chunk = data[offset..offset + size].to_vec();
        offset += size;
        Ok(chunk)
    };
    let positions = take(num_points * 3 * if uses_float16 { 2 } else { 3 })?;
    let alphas = take(num_points)?;
    let colors = take(num_points * 3)?;
    let scales = take(num_points * 3)?;
    let rotations = take(num_points * if uses_quaternion_smallest_three { 4 } else { 3 })?;
    let sh = take(num_points * sh_dim * 3)?;
    Ok(PackedGaussians {
        version,
        num_points,
        sh_degree,
        fractional_bits,
        antialiased: flags & FLAG_ANTIALIASED != 0,
        uses_quaternion_smallest_three,
        positions,
        scales,
        rotations,
        alphas,
        colors,
        sh,
    })
}

/// Reads the packed Gaussians from the bytes of an `.spz` file of either format.
pub fn load_spz_packed_from_bytes(data: &[u8]) -> Result<PackedGaussians> {
    let packed = if data.len() >= 4 && read_u32(data, 0) == NGSP_MAGIC {
        load_ngsp(data)?
    } else if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
        load_legacy(data)?
    } else {
        bail!("spz: unrecognized format");
    };
    packed.check_sizes()?;
    Ok(packed)
}

/// Reads the packed Gaussians of an `.spz` file.
pub fn load_spz_packed(path: impl AsRef<Path>) -> Result<PackedGaussians> {
    let path = path.as_ref();
    let data = std::fs::read(path).with_context(|| format!("spz: cannot read {}", path.display()))?;
    load_spz_packed_from_bytes(&data).with_context(|| format!("spz: cannot load {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rub_to_rdb_flips_y() {
        let c = coordinate_converter(CoordinateSystem::Rub, CoordinateSystem::Rdb);
        assert_eq!(c.flip_p, [1.0, -1.0, 1.0]);
        assert_eq!(c.flip_q, [-1.0, 1.0, -1.0]);
        assert_eq!(c.flip_sh[0], -1.0);
        assert_eq!(c.flip_sh[1], 1.0);
    }

    #[test]
    fn unspecified_converts_nothing() {
        let c = coordinate_converter(CoordinateSystem::Unspecified, CoordinateSystem::Rdb);
        assert_eq!(c, CoordinateConverter::default());
    }

    #[test]
    fn half_to_float_decodes_normals_subnormals_and_infinity() {
        assert_eq!(half_to_float(0x3c00), 1.0);
        assert_eq!(half_to_float(0xc000), -2.0);
        assert_eq!(half_to_float(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_float(0x7c00), f32::INFINITY);
    }

    #[test]
    fn smallest_three_round_trips_the_identity() {
        // Largest component w (index 3) and all others zero: comp = 3 << 30.
        let comp: u32 = 3 << 30;
        let mut q = [0.0; 4];
        unpack_quaternion_smallest_three(&mut q, &comp.to_le_bytes(), &CoordinateConverter::default());
        assert_eq!(q, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_unknown_formats() {
        assert!(load_spz_packed_from_bytes(b"not a splat file").is_err());
    }
}
