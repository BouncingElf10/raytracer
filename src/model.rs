use glam::Vec3;
use crate::color::Color;
use crate::material::Material;
use crate::objects::Triangle;

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
    pub faces: Vec<Face>
}

impl Mesh {
    pub fn new() -> Self {
        Self { faces: Vec::new() }
    }
    pub fn append_face(&mut self, face: Face) {
        self.faces.push(face);
    }
    pub fn get_triangles(&self) -> Vec<Triangle> {
        self.faces
            .iter()
            .flat_map(|face| face.to_tris())
            .collect()
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
        vec![]
    }
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }
}