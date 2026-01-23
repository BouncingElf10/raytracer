use glam::Vec3;
use crate::color::Color;

#[derive(Clone, Copy)]
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
}