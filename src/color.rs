use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Copy)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn newFromNormals(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: (r + 1.0) * 0.5,
            g: (g + 1.0) * 0.5,
            b: (b + 1.0) * 0.5
        }
    }

    pub fn gamma_correct(self) -> Self {
        Self {
            r: self.r.sqrt(),
            g: self.g.sqrt(),
            b: self.b.sqrt(),
        }
    }

    pub fn to_u32(&self) -> u32 {
        ((self.r * 255.0) as u32) << 16 | ((self.g * 255.0) as u32) << 8 | (self.b * 255.0) as u32
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Color::new(self.r.clamp(min, max), self.g.clamp(min, max), self.b.clamp(min, max))
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

pub fn lerp(color1: &Color, color2: &Color, t: f32) -> Color {
    color1.mul(1.0 - t) + color2.mul(t)
}

impl Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b)
    }
}

impl Mul<f32> for Color {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

impl Div<f32> for Color {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self::new(self.r / rhs, self.g / rhs, self.b / rhs)
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Color {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign<f32> for Color {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl MulAssign for Color {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

