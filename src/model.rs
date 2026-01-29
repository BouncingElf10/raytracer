use glam::{Quat, Vec3};
use crate::color::Color;
use crate::material::Material;
use crate::objects::Triangle;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3
}

#[derive(Debug, Clone)]
pub struct Face {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) material: Material
}

pub struct Mesh {
    pub faces: Vec<Face>,
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: f32,
}

impl Mesh {
    pub fn new() -> Self {
        Self { faces: Vec::new(), position: Vec3::ZERO, rotation: Vec3::ZERO, scale: 1.0 }
    }
    pub fn append_face(&mut self, face: Face) {
        self.faces.push(face);
    }

    pub fn get_triangles(&self) -> Vec<Triangle> {
        let rotation =
            Quat::from_rotation_x(self.rotation.x) *
            Quat::from_rotation_y(self.rotation.y) *
            Quat::from_rotation_z(self.rotation.z);

        self.faces
            .iter()
            .flat_map(|face| {
                face.to_tris().into_iter().map(|triangle| {
                    let mut vertices = triangle.get_vertices();
                    for vertex in vertices.iter_mut() {
                        *vertex *= self.scale;
                        *vertex = rotation * *vertex;
                        *vertex += self.position;
                    }
                    Triangle::new(vertices[0], vertices[1], vertices[2], triangle.material().clone())
                })
            }).collect()
    }
}

impl Face {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), material: Material::new(Color::white(), 0.0, 0.0, 0.0)}
    }
    pub fn append_vertex(&mut self, vertex: Vertex) {
        self.vertices.push(vertex);
    }
    pub fn set_material(&mut self, material: Material) {
        self.material = material;
    }
    pub fn to_tris(&self) -> Vec<Triangle> {
        if self.vertices.len() < 3 { return vec![] }
        if self.vertices.len() == 3 {
            return vec![Triangle::new(self.vertices[0].position, self.vertices[1].position, self.vertices[2].position, self.material)]
        }
        let mut tris = Vec::new();
        for i in 1..(self.vertices.len() - 1) {
            tris.push(Triangle::new(self.vertices[0].position, self.vertices[i].position, self.vertices[i + 1].position, self.material, ));
        }
        tris
    }
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }
}