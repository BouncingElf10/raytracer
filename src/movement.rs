use minifb::Key;
use crate::camera::Camera;
use crate::window::Canvas;

pub fn apply_movements(camera: &mut Camera, canvas: &Canvas, delta_time: f32) {
    let mut ray = camera.ray();
    
    let move_speed = 5.0;

    if canvas.is_key_pressed(Key::W) { ray.move_origin_along_direction(move_speed * delta_time); }
    if canvas.is_key_pressed(Key::S) { ray.move_origin_along_direction(-move_speed * delta_time); }

    let rot_speed = 1.5;
    
    if canvas.is_key_pressed(Key::Left) { ray.rotate_y(-rot_speed * delta_time); }
    if canvas.is_key_pressed(Key::Right) { ray.rotate_y(rot_speed * delta_time); }
    if canvas.is_key_pressed(Key::Up) { ray.rotate_x(-rot_speed * delta_time); }
    if canvas.is_key_pressed(Key::Down) { ray.rotate_x(rot_speed * delta_time); }

    camera.set_ray(ray);
}
