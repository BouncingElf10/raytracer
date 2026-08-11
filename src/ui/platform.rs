//! GLFW to Dear ImGui input bridge.
//!
//! The published `imgui-glfw-support` crate targets a much older imgui and glfw
//! than this project uses, so the binding is done here instead. It is small: feed
//! window events into imgui's event queue, keep the display size and frame delta
//! in sync, and report whether imgui wants exclusive use of the mouse or keyboard
//! so the camera controls can stand down while a panel is focused.

use std::time::Instant;

use glfw::{Action, Modifiers, WindowEvent};
use imgui::{Context, Key, MouseButton};

pub struct Platform {
    last_frame: Instant,
    /// Latest cursor position, resent whenever the window regains focus.
    cursor: [f32; 2],
}

impl Platform {
    pub fn new(imgui: &mut Context, width: u32, height: u32, scale: f32) -> Self {
        let io = imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.font_global_scale = scale;

        Self { last_frame: Instant::now(), cursor: [0.0, 0.0] }
    }

    /// Advances the frame clock and refreshes the display size.
    pub fn prepare_frame(&mut self, imgui: &mut Context, width: u32, height: u32) {
        let now = Instant::now();
        let io = imgui.io_mut();
        io.update_delta_time(now - self.last_frame);
        self.last_frame = now;
        io.display_size = [width.max(1) as f32, height.max(1) as f32];
    }

    pub fn handle_event(&mut self, imgui: &mut Context, event: &WindowEvent) {
        let io = imgui.io_mut();

        match event {
            WindowEvent::CursorPos(x, y) => {
                self.cursor = [*x as f32, *y as f32];
                io.add_mouse_pos_event(self.cursor);
            }
            WindowEvent::MouseButton(button, action, _) => {
                if let Some(button) = map_mouse_button(*button) {
                    io.add_mouse_button_event(button, *action != Action::Release);
                }
            }
            WindowEvent::Scroll(dx, dy) => {
                io.add_mouse_wheel_event([*dx as f32, *dy as f32]);
            }
            WindowEvent::Char(character) => {
                io.add_input_character(*character);
            }
            WindowEvent::Key(key, _, action, modifiers) => {
                // Repeats matter for held arrow keys and backspace in text fields.
                let down = *action != Action::Release;
                apply_modifiers(io, *modifiers);
                if let Some(key) = map_key(*key) {
                    io.add_key_event(key, down);
                }
            }
            WindowEvent::Focus(false) => {
                // Buttons held when focus is lost never send a release, so clear
                // them or imgui thinks the mouse is stuck down.
                for button in MouseButton::VARIANTS {
                    io.add_mouse_button_event(button, false);
                }
            }
            WindowEvent::Focus(true) => {
                io.add_mouse_pos_event(self.cursor);
            }
            WindowEvent::Size(width, height) => {
                io.display_size = [(*width).max(1) as f32, (*height).max(1) as f32];
            }
            _ => {}
        }
    }
}

fn apply_modifiers(io: &mut imgui::Io, modifiers: Modifiers) {
    io.add_key_event(Key::ModShift, modifiers.contains(Modifiers::Shift));
    io.add_key_event(Key::ModCtrl, modifiers.contains(Modifiers::Control));
    io.add_key_event(Key::ModAlt, modifiers.contains(Modifiers::Alt));
    io.add_key_event(Key::ModSuper, modifiers.contains(Modifiers::Super));
}

fn map_mouse_button(button: glfw::MouseButton) -> Option<MouseButton> {
    match button {
        glfw::MouseButton::Button1 => Some(MouseButton::Left),
        glfw::MouseButton::Button2 => Some(MouseButton::Right),
        glfw::MouseButton::Button3 => Some(MouseButton::Middle),
        glfw::MouseButton::Button4 => Some(MouseButton::Extra1),
        glfw::MouseButton::Button5 => Some(MouseButton::Extra2),
        _ => None,
    }
}

fn map_key(key: glfw::Key) -> Option<Key> {
    use glfw::Key as G;

    Some(match key {
        G::Tab => Key::Tab,
        G::Left => Key::LeftArrow,
        G::Right => Key::RightArrow,
        G::Up => Key::UpArrow,
        G::Down => Key::DownArrow,
        G::PageUp => Key::PageUp,
        G::PageDown => Key::PageDown,
        G::Home => Key::Home,
        G::End => Key::End,
        G::Insert => Key::Insert,
        G::Delete => Key::Delete,
        G::Backspace => Key::Backspace,
        G::Space => Key::Space,
        G::Enter => Key::Enter,
        G::Escape => Key::Escape,
        G::LeftControl => Key::LeftCtrl,
        G::LeftShift => Key::LeftShift,
        G::LeftAlt => Key::LeftAlt,
        G::LeftSuper => Key::LeftSuper,
        G::RightControl => Key::RightCtrl,
        G::RightShift => Key::RightShift,
        G::RightAlt => Key::RightAlt,
        G::RightSuper => Key::RightSuper,
        G::Menu => Key::Menu,
        G::Num0 => Key::Alpha0,
        G::Num1 => Key::Alpha1,
        G::Num2 => Key::Alpha2,
        G::Num3 => Key::Alpha3,
        G::Num4 => Key::Alpha4,
        G::Num5 => Key::Alpha5,
        G::Num6 => Key::Alpha6,
        G::Num7 => Key::Alpha7,
        G::Num8 => Key::Alpha8,
        G::Num9 => Key::Alpha9,
        G::A => Key::A, G::B => Key::B, G::C => Key::C, G::D => Key::D,
        G::E => Key::E, G::F => Key::F, G::G => Key::G, G::H => Key::H,
        G::I => Key::I, G::J => Key::J, G::K => Key::K, G::L => Key::L,
        G::M => Key::M, G::N => Key::N, G::O => Key::O, G::P => Key::P,
        G::Q => Key::Q, G::R => Key::R, G::S => Key::S, G::T => Key::T,
        G::U => Key::U, G::V => Key::V, G::W => Key::W, G::X => Key::X,
        G::Y => Key::Y, G::Z => Key::Z,
        G::F1 => Key::F1, G::F2 => Key::F2, G::F3 => Key::F3, G::F4 => Key::F4,
        G::F5 => Key::F5, G::F6 => Key::F6, G::F7 => Key::F7, G::F8 => Key::F8,
        G::F9 => Key::F9, G::F10 => Key::F10, G::F11 => Key::F11, G::F12 => Key::F12,
        G::Apostrophe => Key::Apostrophe,
        G::Comma => Key::Comma,
        G::Minus => Key::Minus,
        G::Period => Key::Period,
        G::Slash => Key::Slash,
        G::Semicolon => Key::Semicolon,
        G::Equal => Key::Equal,
        G::LeftBracket => Key::LeftBracket,
        G::Backslash => Key::Backslash,
        G::RightBracket => Key::RightBracket,
        G::GraveAccent => Key::GraveAccent,
        G::CapsLock => Key::CapsLock,
        G::ScrollLock => Key::ScrollLock,
        G::NumLock => Key::NumLock,
        G::PrintScreen => Key::PrintScreen,
        G::Pause => Key::Pause,
        G::Kp0 => Key::Keypad0, G::Kp1 => Key::Keypad1, G::Kp2 => Key::Keypad2,
        G::Kp3 => Key::Keypad3, G::Kp4 => Key::Keypad4, G::Kp5 => Key::Keypad5,
        G::Kp6 => Key::Keypad6, G::Kp7 => Key::Keypad7, G::Kp8 => Key::Keypad8,
        G::Kp9 => Key::Keypad9,
        G::KpDecimal => Key::KeypadDecimal,
        G::KpDivide => Key::KeypadDivide,
        G::KpMultiply => Key::KeypadMultiply,
        G::KpSubtract => Key::KeypadSubtract,
        G::KpAdd => Key::KeypadAdd,
        G::KpEnter => Key::KeypadEnter,
        G::KpEqual => Key::KeypadEqual,
        _ => return None,
    })
}
