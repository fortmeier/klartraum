// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! The few [glm](https://github.com/g-truc/glm) types and functions the engine uses.
//!
//! They follow glm's formulas operation by operation (column-major matrices,
//! right-handed `look_at`, OpenGL depth range in `perspective`), so the camera
//! uniform buffer gets the same values as in the C++ engine. Where glm writes a
//! scalar `a * b + c` in one expression, clang contracts it into a fused
//! multiply-add (the default `-ffp-contract=on`, used on arm64), and so do these
//! functions with `mul_add`; glm's vector-operator chains are not contracted.

use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use bytemuck::{Pod, Zeroable};

/// A 3-component vector, as `glm::vec3`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A 4-component vector, as `glm::vec4`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// A column-major 4x4 matrix, as `glm::mat4`; `m[c][r]` is column `c`, row `r`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}

pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

pub const fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4 { x, y, z, w }
}

impl Vec3 {
    pub const ZERO: Vec3 = vec3(0.0, 0.0, 0.0);

    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(a: [f32; 3]) -> Self {
        vec3(a[0], a[1], a[2])
    }
}

impl Vec4 {
    pub fn splat(v: f32) -> Self {
        vec4(v, v, v, v)
    }

    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    pub fn xyz(self) -> Vec3 {
        vec3(self.x, self.y, self.z)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index {i} out of range"),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index {i} out of range"),
        }
    }
}

macro_rules! componentwise {
    ($t:ident { $($f:ident),* }) => {
        impl Add for $t {
            type Output = $t;
            fn add(self, o: $t) -> $t { $t { $($f: self.$f + o.$f),* } }
        }
        impl Sub for $t {
            type Output = $t;
            fn sub(self, o: $t) -> $t { $t { $($f: self.$f - o.$f),* } }
        }
        impl Mul for $t {
            type Output = $t;
            fn mul(self, o: $t) -> $t { $t { $($f: self.$f * o.$f),* } }
        }
        impl Mul<f32> for $t {
            type Output = $t;
            fn mul(self, s: f32) -> $t { $t { $($f: self.$f * s),* } }
        }
        impl Mul<$t> for f32 {
            type Output = $t;
            fn mul(self, v: $t) -> $t { $t { $($f: self * v.$f),* } }
        }
        impl Neg for $t {
            type Output = $t;
            fn neg(self) -> $t { $t { $($f: -self.$f),* } }
        }
    };
}

componentwise!(Vec3 { x, y, z });
componentwise!(Vec4 { x, y, z, w });

impl Index<usize> for Mat4 {
    type Output = Vec4;
    fn index(&self, c: usize) -> &Vec4 {
        &self.cols[c]
    }
}

impl IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, c: usize) -> &mut Vec4 {
        &mut self.cols[c]
    }
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4::diagonal(1.0);

    /// `glm::mat4(s)`: `s` on the diagonal.
    pub const fn diagonal(s: f32) -> Mat4 {
        Mat4 {
            cols: [
                vec4(s, 0.0, 0.0, 0.0),
                vec4(0.0, s, 0.0, 0.0),
                vec4(0.0, 0.0, s, 0.0),
                vec4(0.0, 0.0, 0.0, s),
            ],
        }
    }

    pub fn to_cols_array(&self) -> [f32; 16] {
        bytemuck::cast(*self)
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, m2: Mat4) -> Mat4 {
        let [a0, a1, a2, a3] = self.cols;
        let col = |b: Vec4| a0 * b.x + a1 * b.y + a2 * b.z + a3 * b.w;
        Mat4 {
            cols: [col(m2[0]), col(m2[1]), col(m2[2]), col(m2[3])],
        }
    }
}

impl Mul<f32> for Mat4 {
    type Output = Mat4;
    fn mul(self, s: f32) -> Mat4 {
        Mat4 {
            cols: self.cols.map(|c| c * s),
        }
    }
}

/// `glm::dot` for vec3: the products are formed first, then summed.
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    let tmp = a * b;
    tmp.x + tmp.y + tmp.z
}

/// `glm::cross`; `a * b - c * d` is contracted to `fma(a, b, -(c * d))` as clang does.
pub fn cross(x: Vec3, y: Vec3) -> Vec3 {
    vec3(
        x.y.mul_add(y.z, -(y.y * x.z)),
        x.z.mul_add(y.x, -(y.z * x.x)),
        x.x.mul_add(y.y, -(y.x * x.y)),
    )
}

/// `glm::inversesqrt`.
pub fn inversesqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// `glm::normalize`.
pub fn normalize(v: Vec3) -> Vec3 {
    v * inversesqrt(dot(v, v))
}

/// `glm::radians`.
pub fn radians(degrees: f32) -> f32 {
    degrees * 0.01745329251994329576923690768489f32
}

/// `glm::translate(m, v)`.
pub fn translate(m: Mat4, v: Vec3) -> Mat4 {
    let mut result = m;
    result[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
    result
}

/// `glm::rotate(m, angle, axis)`: `m` followed by a rotation of `angle` radians about `v`.
pub fn rotate(m: Mat4, angle: f32, v: Vec3) -> Mat4 {
    let a = angle;
    let c = a.cos();
    let s = a.sin();

    let axis = normalize(v);
    let temp = (1.0 - c) * axis;

    let mut r = Mat4::default();
    r[0][0] = temp[0].mul_add(axis[0], c);
    r[0][1] = temp[0].mul_add(axis[1], s * axis[2]);
    r[0][2] = temp[0].mul_add(axis[2], -(s * axis[1]));

    r[1][0] = temp[1].mul_add(axis[0], -(s * axis[2]));
    r[1][1] = temp[1].mul_add(axis[1], c);
    r[1][2] = temp[1].mul_add(axis[2], s * axis[0]);

    r[2][0] = temp[2].mul_add(axis[0], s * axis[1]);
    r[2][1] = temp[2].mul_add(axis[1], -(s * axis[0]));
    r[2][2] = temp[2].mul_add(axis[2], c);

    let mut result = Mat4::default();
    result[0] = m[0] * r[0][0] + m[1] * r[0][1] + m[2] * r[0][2];
    result[1] = m[0] * r[1][0] + m[1] * r[1][1] + m[2] * r[1][2];
    result[2] = m[0] * r[2][0] + m[1] * r[2][1] + m[2] * r[2][2];
    result[3] = m[3];
    result
}

/// `glm::lookAt` (right-handed).
pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = normalize(center - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    let mut result = Mat4::IDENTITY;
    result[0][0] = s.x;
    result[1][0] = s.y;
    result[2][0] = s.z;
    result[0][1] = u.x;
    result[1][1] = u.y;
    result[2][1] = u.z;
    result[0][2] = -f.x;
    result[1][2] = -f.y;
    result[2][2] = -f.z;
    result[3][0] = -dot(s, eye);
    result[3][1] = -dot(u, eye);
    result[3][2] = dot(f, eye);
    result
}

/// `glm::perspective` (right-handed, OpenGL depth range -1 to 1).
pub fn perspective(fovy: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let tan_half_fovy = (fovy / 2.0).tan();

    let mut result = Mat4::diagonal(0.0);
    result[0][0] = 1.0 / (aspect * tan_half_fovy);
    result[1][1] = 1.0 / tan_half_fovy;
    result[2][2] = -(z_far + z_near) / (z_far - z_near);
    result[2][3] = -1.0;
    result[3][2] = -(2.0 * z_far * z_near) / (z_far - z_near);
    result
}

/// `glm::inverse` for a 4x4 matrix (cofactors and determinant, as glm computes them).
pub fn inverse(m: Mat4) -> Mat4 {
    let d = |a: f32, b: f32, c: f32, e: f32| a.mul_add(b, -(c * e));
    let coef00 = d(m[2][2], m[3][3], m[3][2], m[2][3]);
    let coef02 = d(m[1][2], m[3][3], m[3][2], m[1][3]);
    let coef03 = d(m[1][2], m[2][3], m[2][2], m[1][3]);

    let coef04 = d(m[2][1], m[3][3], m[3][1], m[2][3]);
    let coef06 = d(m[1][1], m[3][3], m[3][1], m[1][3]);
    let coef07 = d(m[1][1], m[2][3], m[2][1], m[1][3]);

    let coef08 = d(m[2][1], m[3][2], m[3][1], m[2][2]);
    let coef10 = d(m[1][1], m[3][2], m[3][1], m[1][2]);
    let coef11 = d(m[1][1], m[2][2], m[2][1], m[1][2]);

    let coef12 = d(m[2][0], m[3][3], m[3][0], m[2][3]);
    let coef14 = d(m[1][0], m[3][3], m[3][0], m[1][3]);
    let coef15 = d(m[1][0], m[2][3], m[2][0], m[1][3]);

    let coef16 = d(m[2][0], m[3][2], m[3][0], m[2][2]);
    let coef18 = d(m[1][0], m[3][2], m[3][0], m[1][2]);
    let coef19 = d(m[1][0], m[2][2], m[2][0], m[1][2]);

    let coef20 = d(m[2][0], m[3][1], m[3][0], m[2][1]);
    let coef22 = d(m[1][0], m[3][1], m[3][0], m[1][1]);
    let coef23 = d(m[1][0], m[2][1], m[2][0], m[1][1]);

    let fac0 = vec4(coef00, coef00, coef02, coef03);
    let fac1 = vec4(coef04, coef04, coef06, coef07);
    let fac2 = vec4(coef08, coef08, coef10, coef11);
    let fac3 = vec4(coef12, coef12, coef14, coef15);
    let fac4 = vec4(coef16, coef16, coef18, coef19);
    let fac5 = vec4(coef20, coef20, coef22, coef23);

    let vec0 = vec4(m[1][0], m[0][0], m[0][0], m[0][0]);
    let vec1 = vec4(m[1][1], m[0][1], m[0][1], m[0][1]);
    let vec2 = vec4(m[1][2], m[0][2], m[0][2], m[0][2]);
    let vec3_ = vec4(m[1][3], m[0][3], m[0][3], m[0][3]);

    let inv0 = vec1 * fac0 - vec2 * fac1 + vec3_ * fac2;
    let inv1 = vec0 * fac0 - vec2 * fac3 + vec3_ * fac4;
    let inv2 = vec0 * fac1 - vec1 * fac3 + vec3_ * fac5;
    let inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

    let sign_a = vec4(1.0, -1.0, 1.0, -1.0);
    let sign_b = vec4(-1.0, 1.0, -1.0, 1.0);
    let inverse = Mat4 {
        cols: [inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b],
    };

    let row0 = vec4(inverse[0][0], inverse[1][0], inverse[2][0], inverse[3][0]);

    let dot0 = m[0] * row0;
    let dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);

    let one_over_determinant = 1.0 / dot1;

    inverse * one_over_determinant
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Mat4, b: Mat4) {
        for (x, y) in a.to_cols_array().iter().zip(b.to_cols_array()) {
            assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn layout_is_column_major_and_packed() {
        assert_eq!(std::mem::size_of::<Mat4>(), 64);
        let m = translate(Mat4::IDENTITY, vec3(1.0, 2.0, 3.0));
        assert_eq!(&m.to_cols_array()[12..15], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn inverse_undoes_a_transform() {
        let m = translate(rotate(Mat4::IDENTITY, 0.7, vec3(0.3, 1.0, 0.2)), vec3(1.0, -2.0, 0.5));
        assert_close(m * inverse(m), Mat4::IDENTITY);
    }

    #[test]
    fn rotate_quarter_turn_about_z_maps_x_to_y() {
        let m = rotate(Mat4::IDENTITY, std::f32::consts::FRAC_PI_2, vec3(0.0, 0.0, 1.0));
        let x = m[0];
        assert!(x.x.abs() < 1e-6 && (x.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn look_at_puts_the_eye_at_the_origin() {
        let eye = vec3(1.0, 2.0, 3.0);
        let m = look_at(eye, Vec3::ZERO, vec3(0.0, 1.0, 0.0));
        let p = m[0] * eye.x + m[1] * eye.y + m[2] * eye.z + m[3];
        assert!(p.x.abs() < 1e-6 && p.y.abs() < 1e-6 && p.z.abs() < 1e-6);
    }
}
