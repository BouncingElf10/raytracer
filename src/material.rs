use glam::Vec3;
use crate::color::Color;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub albedo: Color,
    pub roughness: f32,
    pub metallic: f32,
    pub emission: f32
}

impl Material {
    pub fn new(albedo: Color, roughness: f32, metallic: f32, emission: f32) -> Self {
        Self { albedo, roughness, metallic, emission }
    }
    
    pub fn default() -> Self {
        Self { albedo: Color::new(0.5, 0.5, 0.5), roughness: 0.0, metallic: 0.0, emission: 0.0 }
    }
}