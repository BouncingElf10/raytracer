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

    pub fn to_u32(&self) -> u32 {
        ((self.r * 255.0) as u32) << 16 | ((self.g * 255.0) as u32) << 8 | (self.b * 255.0) as u32
    }
}