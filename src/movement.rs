use minifb::Key;
use glam::Vec3;
use crate::camera::Camera;
use crate::window::Canvas;

pub struct MovementState {
    last_mouse_x: f32,
    last_mouse_y: f32,
    first_mouse: bool,
    yaw: f32,
    pitch: f32,
}

impl MovementState {
    pub fn new() -> Self {
        Self {
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            first_mouse: true,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

pub fn apply_movements(camera: &mut Camera, canvas: &Canvas, delta_time: f32, state: &mut MovementState) {
    let mut ray = camera.ray();
    if let Some((mouse_x, mouse_y)) = canvas.get_mouse_pos() {
        if state.first_mouse {
            state.last_mouse_x = mouse_x;
            state.last_mouse_y = mouse_y;
            state.first_mouse = false;
        }

        let x_offset = state.last_mouse_x - mouse_x;
        let y_offset = state.last_mouse_y - mouse_y;

        state.last_mouse_x = mouse_x;
        state.last_mouse_y = mouse_y;

        let sensitivity = 0.1;
        let x_offset = x_offset * sensitivity;
        let y_offset = y_offset * sensitivity;

        state.yaw += x_offset;
        state.pitch += y_offset;

        if state.pitch > 89.0 {
            state.pitch = 89.0;
        }
        if state.pitch < -89.0 {
            state.pitch = -89.0;
        }

        let direction = Vec3::new(
            state.yaw.to_radians().cos() * state.pitch.to_radians().cos(),
            state.pitch.to_radians().sin(),
            state.yaw.to_radians().sin() * state.pitch.to_radians().cos(),
        ).normalize();

        ray = crate::ray::Ray::new(ray.origin(), direction);
    }

    let move_speed = 5.0;

    let forward = ray.direction().normalize();
    let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).normalize();
    let up = Vec3::new(0.0, 1.0, 0.0);

    let mut movement = Vec3::ZERO;

    if canvas.is_key_down(Key::W) {
        movement += forward * move_speed * delta_time;
    }
    if canvas.is_key_down(Key::S) {
        movement -= forward * move_speed * delta_time;
    }
    if canvas.is_key_down(Key::A) {
        movement -= right * move_speed * delta_time;
    }
    if canvas.is_key_down(Key::D) {
        movement += right * move_speed * delta_time;
    }
    if canvas.is_key_down(Key::Space) {
        movement += up * move_speed * delta_time;
    }
    if canvas.is_key_down(Key::LeftShift) {
        movement -= up * move_speed * delta_time;
    }

    ray = crate::ray::Ray::new(ray.origin() + movement, ray.direction());

    camera.set_ray(ray);
}