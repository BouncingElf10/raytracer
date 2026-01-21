use minifb::{Key, KeyRepeat, Window, WindowOptions};

pub struct Canvas {
    width: usize,
    height: usize,
    buffer: Vec<u32>,
    window: Window,
}

impl Canvas {
    pub fn new(width: usize, height: usize, title: &str) -> Self {
        let window = Window::new(title, width, height,
            WindowOptions {
                resize: false,
                ..WindowOptions::default()
            },
        ).expect("Failed to create window");

        let buffer = vec![0; width * height];

        Self { width, height, buffer, window }
    }

    pub fn paint_pixel(&mut self, x: usize, y: usize, color: u32) {
        if x < self.width && y < self.height {
            self.buffer[y * self.width + x] = color;
        }
    }

    pub fn update(&mut self) {
        self.window
            .update_with_buffer(&self.buffer, self.width, self.height)
            .unwrap();
    }

    pub fn set_window_title(&mut self, title: &str) {
        self.window.set_title(title);
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    
    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.window.is_key_pressed(key, KeyRepeat::Yes)
    }
    
    pub fn clear(&mut self) {
        self.buffer.fill(0);
    }
}
