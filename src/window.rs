use crate::color::Color;
use glfw::{fail_on_errors, Action, Context, CursorMode, Glfw, GlfwReceiver, Key, PWindow, WindowEvent};
use wgpu::TextureUsages;

#[allow(dead_code)]
pub struct Canvas {
    width: u32,
    height: u32,
    window: PWindow,
    events: GlfwReceiver<(f64, WindowEvent)>,
    glfw: Glfw,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pixel_buffer: Vec<u32>,
    pixel_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
    pub(crate) accum_buffer: Vec<Color>,
    pub(crate) sample_count: u32,
    pub compute_pipeline: Option<wgpu::ComputePipeline>,
    pub compute_bind_group: Option<wgpu::BindGroup>,
    pub sphere_buffer: Option<wgpu::Buffer>,
    pub triangle_buffer: Option<wgpu::Buffer>,
    pub plane_buffer: Option<wgpu::Buffer>,
    pub ray_buffer: Option<wgpu::Buffer>,
    pub hit_buffer: Option<wgpu::Buffer>,
    pub staging_buffer: Option<wgpu::Buffer>,
    pub counts_buffer: Option<wgpu::Buffer>,
}

impl Canvas {
    pub async fn new(width: u32, height: u32, title: &str) -> Self {
        let mut glfw = glfw::init(fail_on_errors).expect("Failed to initialize GLFW");

        let (mut window, events) = glfw
            .create_window(width, height, title, glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window");

        window.set_all_polling(true);
        window.make_current();
        window.set_cursor_mode(CursorMode::Disabled);

        let accum_buffer = vec![Color::black(); (width * height) as usize];
        let sample_count = 0;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe { instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap())
                .expect("Failed to create surface")
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: Default::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: Default::default(),
                },
            ).await.expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let pixel_buffer = vec![0u32; (width * height) as usize];

        let pixel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pixel Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = pixel_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Screen Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/screen.wgsl").into()
            ),
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Screen Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Screen Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            });

        Self { width, height, window, events, glfw, surface, device, queue, config, pixel_buffer,
            pixel_texture, bind_group, bind_group_layout, render_pipeline, accum_buffer, sample_count,
            compute_pipeline: None, compute_bind_group: None, sphere_buffer: None, triangle_buffer: None,
            plane_buffer: None, ray_buffer: None, hit_buffer: None, staging_buffer: None, counts_buffer: None }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);

            self.pixel_buffer = vec![0u32; (width * height) as usize];
            self.pixel_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Pixel Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });

            self.accum_buffer = vec![Color::black(); (width * height) as usize];
            self.sample_count = 0;
        }
    }

    fn render(&mut self, clear_color: wgpu::Color) -> Result<(), wgpu::SurfaceError> {
        let rgba_data: Vec<u8> = self.pixel_buffer
            .iter()
            .flat_map(|&color| {
                [
                    ((color >> 16) & 0xFF) as u8,
                    ((color >> 8) & 0xFF) as u8,
                    (color & 0xFF) as u8,
                    ((color >> 24) & 0xFF) as u8,
                ]
            }).collect();

        self.queue.write_texture(
            self.pixel_texture.as_image_copy(),
            &rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder"), });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                self.window.set_should_close(true);
            }
            WindowEvent::Size(width, height) => {
                self.resize(width as u32, height as u32);
            }
            _ => {}
        }
    }

    pub fn paint_pixel(&mut self, x: u32, y: u32, color: u32) {
        if x < self.width && y < self.height {
            let index = (y * self.width + x) as usize;
            self.pixel_buffer[index] = color;
        }
    }

    pub fn update(&mut self) {
        self.glfw.poll_events();
        let events: Vec<_> = glfw::flush_messages(&self.events).collect();
        for (_, event) in events {
            self.handle_event(event);
        }
    }

    pub fn present(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.render(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0, })
    }

    pub fn reset_accumulation(&mut self) {
        self.accum_buffer.fill(Color::black());
        self.sample_count = 0;
    }

    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }
    
    pub fn is_open(&self) -> bool {
        !self.window.should_close()
    }

    pub fn set_window_title(&mut self, title: &str) {
        self.window.set_title(title);
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn is_key_down(&self, key: Key) -> bool {
        self.window.get_key(key) == Action::Press
    }

    pub fn get_mouse_pos(&self) -> (f64, f64) {
        self.window.get_cursor_pos()
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}