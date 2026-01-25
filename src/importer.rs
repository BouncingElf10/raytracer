use std::fs::File;
use std::io::Read;
use glam::Vec3;
use crate::model::{Face, Mesh, Vertex};

pub fn import_obj(path: &str) -> Mesh {
    let file = File::open(path);
    let mut contents = String::new();
    file.expect("Failed to open file").read_to_string(&mut contents).expect("Failed to read string!");
    parse_obj_into_mesh(contents)
}

fn parse_obj_into_mesh(file_str: String) -> Mesh {
    let mut mesh = Mesh::new();
    
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();

    file_str.lines().for_each(|line| {
        let line = line.trim();
        
        if line.is_empty() || line.starts_with('#') {
            return;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        match parts[0] {
            "v" => {
                if parts.len() >= 4 {
                    let x = parts[1].parse::<f32>().unwrap_or(0.0);
                    let y = parts[2].parse::<f32>().unwrap_or(0.0);
                    let z = parts[3].parse::<f32>().unwrap_or(0.0);
                    positions.push(Vec3::new(x, y, z));
                }
            },
            "vn" => {
                if parts.len() >= 4 {
                    let x = parts[1].parse::<f32>().unwrap_or(0.0);
                    let y = parts[2].parse::<f32>().unwrap_or(0.0);
                    let z = parts[3].parse::<f32>().unwrap_or(0.0);
                    normals.push(Vec3::new(x, y, z));
                }
            },
            "f" => {
                let mut face = Face::new();

                for i in 1..parts.len() {
                    let indices: Vec<&str> = parts[i].split('/').collect();
                    
                    let pos_idx = indices[0].parse::<usize>().unwrap_or(1) - 1; // OBJ is 1-indexed
                    let position = positions.get(pos_idx).copied().unwrap_or(Vec3::ZERO);
                    
                    let normal = if indices.len() >= 3 && !indices[2].is_empty() {
                        let norm_idx = indices[2].parse::<usize>().unwrap_or(1) - 1;
                        normals.get(norm_idx).copied().unwrap_or(Vec3::Y)
                    } else {
                        Vec3::Y
                    };

                    face.append_vertex(Vertex::new(position, normal));
                }

                if !face.vertices.is_empty() {
                    mesh.append_face(face);
                }
            },
            _ => {}
        }
    });

    mesh
}