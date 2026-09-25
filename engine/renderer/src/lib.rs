//! feathered-renderer — wgpu pipelines, atlas binding, animation clock.
//!
//! Phase 1 renders three pipelines (opaque, cutout, translucent) sampling the
//! single block atlas. Water animation is driven by a uniform of frame row
//! offsets indexed by `anim_id` (reserved; Phase 1 water uses frame-0 UVs).
//!
//! API notes (wgpu 30): `create_surface` returns `Result`; acquiring a frame
//! returns the `CurrentSurfaceTexture` enum; presentation goes through
//! `Queue::present(texture)`.

use feathered_assets::atlas::Atlas;
use feathered_chunk::{mesh_world, MeshedChunk, TintPolicy, Vertex};
use feathered_world::grid::World;
use feathered_world::Registry;
use wgpu::util::DeviceExt;

/// Camera: position + yaw/pitch, perspective projection.
pub struct Camera {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// Combined view-projection, built directly in **WGSL storage order**:
    /// `out[j]` is column j of the mathematical matrix M = P·V (so the shader's
    /// `M * v` with column vectors is exact). No transpose/mul ambiguity.
    pub fn view_proj(&self) -> [[f32; 4]; 4] {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        // Pitch sign: NEGATIVE pitch = look down (client's convention), so
        // forward.y = sin_p goes negative. Verified: eye (16,10,30) pitch
        // -0.3 puts the ground-block center at NDC (0, +0.52, 0.92).
        let forward = [sin_y * cos_p, sin_p, -cos_y * cos_p];
        let right = [cos_y, 0.0, sin_y];
        let up = [-sin_y * sin_p, cos_p, cos_y * sin_p];
        let eye = self.pos;

        // View as a row-major operator matrix for column vectors: each ROW
        // is a basis vector (right/up/−forward) plus its translation term.
        // (Basis vectors in COLUMNS would give the inverse rotation.)
        let view = [
            [right[0], right[1], right[2], -dot(right, eye)],
            [up[0], up[1], up[2], -dot(up, eye)],
            [-forward[0], -forward[1], -forward[2], dot(forward, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Perspective, row-major, [0,1] depth range (Vulkan/wgpu):
        //   z_clip = far/(near-far)·z + far·near/(near-far)
        //   w_clip = -z
        // (b = far·near/(near-far) lives in row 2's homogeneous slot; row 3
        // is the w row.) No y-flip: wgpu's framebuffer rows are top-down, so
        // world-up → NDC-up → lower pixel rows renders upright as-is.
        // NOTE Phase 2: when re-enabling back-face culling, verify quad
        // winding against this convention (FrontFace may need to be Cw).
        let f = 1.0 / (self.fov_y / 2.0).tan();
        let (far, near) = (self.far, self.near);
        let proj = [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ];

        // The shader computes N·v with N[i][j] = storage[j][i]; we need
        // N = P·V (operator product). storage[j][i] = N[i][j] =
        // Σ_m proj[i][m]·view[m][j].
        let mut out = [[0.0f32; 4]; 4];
        for j in 0..4 {
            for i in 0..4 {
                let mut acc = 0.0;
                for m in 0..4 {
                    acc += proj[i][m] * view[m][j];
                }
                out[j][i] = acc;
            }
        }
        out
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// One animated sprite's render slot (index in the vec = the mesher's
/// `anim_id - 1`). Timing comes from the pack; `stride` is the vertical
/// distance between frame origins **pre-normalized** by the atlas height so
/// the shader can use it directly.
#[derive(Debug, Clone)]
pub struct AnimSlot {
    pub frametime: u32,
    /// Playback order; empty = sequential over `strip_frames` rows.
    pub frames: Vec<u16>,
    pub strip_frames: u32,
    pub interpolate: bool,
    /// Normalized per-frame v stride (frame_stride_px / atlas_height).
    pub stride: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub anim_offsets: [f32; 64],
}

/// Fragment blend strategy per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Binary alpha (discard < 0.5), no blending — opaque + cutout layers.
    Cutout,
    /// Real alpha blending — translucent layer (water).
    Alpha,
}

/// The renderer. Owns device, pipelines and GPU resources.
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipelines: LayerPipelines,
    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
    anims: Vec<AnimSlot>,
    /// Pipeline set targeting Rgba8UnormSrgb (offscreen captures).
    capture_pipelines: LayerPipelines,
    /// CPU copy of the last captured frame. Not allocated until first capture.
    last_frame: Option<(u32, u32, Vec<u8>)>,
}

struct LayerPipelines {
    opaque: wgpu::RenderPipeline,
    cutout: wgpu::RenderPipeline,
    translucent: wgpu::RenderPipeline,
    atlas_bind: wgpu::BindGroup,
}

impl Renderer {
    /// Create the renderer for a window surface (async; run under pollster).
    /// `anims` is the animation slot table shared with the mesher's `anim_id`.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>> + 'static,
        width: u32,
        height: u32,
        atlas: &Atlas,
        anims: Vec<AnimSlot>,
    ) -> Renderer {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window)
            .expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("no suitable adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("feathered"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");
        let surface_format = surface
            .get_capabilities(&adapter)
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        let surface_config = wgpu::SurfaceConfiguration {
            // COPY_SRC enables the screenshot readback path.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Upload the atlas + mip chain (mip-major ordering).
        let atlas_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("block-atlas"),
                size: wgpu::Extent3d {
                    width: atlas.width,
                    height: atlas.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: atlas.mip_levels(),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::MipMajor,
            &mip_concat(atlas),
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Nearest sampler over the sRGB atlas. ClampToEdge keeps gutter
        // clamping intact; the atlas never tiles by construction.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::bytes_of(&Globals {
                view_proj: [[0.0; 4]; 4],
                anim_offsets: [0.0; 64],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atlas-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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

        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals-bg"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });
        let atlas_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas-bg"),
            layout: &atlas_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("terrain.wgsl").into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pl"),
            bind_group_layouts: &[Some(&globals_bgl), Some(&atlas_bgl)],
            immediate_size: 0,
        });

        let depth = Self::make_depth(&device, width.max(1), height.max(1));
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm16x2, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint8x2, offset: 20, shader_location: 3 },
            ],
        })];

        // Pipeline factory per color format: the surface gets its native
        // format; the offscreen capture target gets Rgba8UnormSrgb (matching
        // gamma so the PNG equals what the window shows).
        let make_pipelines = |fmt: wgpu::TextureFormat| {
            let make_pipeline = |blend: BlendMode| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("terrain"),
                    layout: Some(&pl_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(match blend {
                            BlendMode::Cutout => "fs_main",
                            BlendMode::Alpha => "fs_main_blend",
                        }),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: fmt,
                            blend: Some(match blend {
                                BlendMode::Cutout => wgpu::BlendState::REPLACE,
                                BlendMode::Alpha => wgpu::BlendState::ALPHA_BLENDING,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        // Phase 1: culling disabled until winding is verified
                        // against the y-down framebuffer.
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth24Plus,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        bias: wgpu::DepthBiasState {
                            constant: if blend == BlendMode::Alpha { 2 } else { 0 },
                            slope_scale: if blend == BlendMode::Alpha { 2.0 } else { 0.0 },
                            clamp: 0.0,
                        },
                    }),
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
            LayerPipelines {
                opaque: make_pipeline(BlendMode::Cutout),
                cutout: make_pipeline(BlendMode::Cutout),
                translucent: make_pipeline(BlendMode::Alpha),
                atlas_bind: atlas_bind.clone(),
            }
        };

        let pipelines = make_pipelines(surface_format);
        let capture_pipelines = make_pipelines(wgpu::TextureFormat::Rgba8UnormSrgb);

        Renderer {
            device,
            queue,
            surface,
            surface_config,
            pipelines,
            globals_buf,
            globals_bind,
            depth_view,
            depth_size: (width.max(1), height.max(1)),
            anims,
            capture_pipelines,
            last_frame: None,
        }
    }

    fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        let depth = Self::make_depth(&self.device, width, height);
        self.depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_size = (width, height);
    }

    /// Render one frame.
    pub fn render(&mut self, meshes: &MeshedChunk, camera: &Camera, tick: u64) {
        self.render_inner(meshes, camera, tick);
    }

    /// Render one frame into an offscreen target and keep a CPU copy for
    /// `capture_last_frame` (no present, no vsync — deterministic captures).
    pub fn render_capture(&mut self, meshes: &MeshedChunk, camera: &Camera, tick: u64) {
        self.render_offscreen(meshes, camera, tick);
    }

    /// RGBA bytes (top-down rows) of the last `render_capture` frame.
    pub fn capture_last_frame(&self) -> Result<Vec<u8>, String> {
        self.last_frame
            .as_ref()
            .map(|(_, _, px)| px.clone())
            .ok_or_else(|| "no captured frame yet (call render_capture first)".into())
    }

    /// Pixel dimensions of the last captured frame.
    pub fn frame_size(&self) -> (u32, u32) {
        self.last_frame
            .as_ref()
            .map(|(w, h, _)| (*w, *h))
            .unwrap_or((0, 0))
    }

    /// Animation offsets (uniform row 64 floats) for the current tick.
    fn animation_offsets(&self, tick: u64) -> [f32; 64] {
        // slot 0 is unused (anim_id 0 = static sprite). `stride` arrives
        // pre-normalized (frame_stride_px / atlas_height) from the client.
        let mut anim_offsets = [0.0f32; 64];
        for (slot, a) in self.anims.iter().enumerate() {
            if slot + 1 >= 64 {
                break;
            }
            let seq_len = if a.frames.is_empty() {
                a.strip_frames
            } else {
                a.frames.len() as u32
            };
            if seq_len == 0 {
                continue;
            }
            let idx = ((tick / a.frametime.max(1) as u64) % seq_len as u64) as usize;
            let row = if a.frames.is_empty() {
                idx as u32
            } else {
                a.frames[idx] as u32
            };
            if a.interpolate && seq_len > 1 {
                // Fractional row position: the shader blends the two rows
                // spanned by the fractional offset (Phase 1 nearest-sampler
                // makes this approximate; exact interpolation arrives with the
                // dual-sample water shader).
                let t = (tick % a.frametime.max(1) as u64) as f32 / a.frametime.max(1) as f32;
                anim_offsets[slot + 1] = (row as f32 + t) * a.stride;
            } else {
                anim_offsets[slot + 1] = row as f32 * a.stride;
            }
        }
        anim_offsets
    }

    /// Record the three-layer scene draw into `rpass`.
    fn record_scene(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        meshes: &MeshedChunk,
        pipes: &LayerPipelines,
    ) {
        rpass.set_bind_group(0, &self.globals_bind, &[]);

        // Per-layer buffer upload (Phase 1 keeps it simple; persistent
        // buffers arrive with the chunk-streaming milestone).
        for (pipe, layer) in [
            (&pipes.opaque, &meshes.opaque),
            (&pipes.cutout, &meshes.cutout),
            (&pipes.translucent, &meshes.translucent),
        ] {
            if layer.indices.is_empty() {
                continue;
            }
            let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&layer.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&layer.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            rpass.set_pipeline(pipe);
            rpass.set_bind_group(1, &self.pipelines.atlas_bind, &[]);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..layer.indices.len() as u32, 0, 0..1);
        }
    }

    const SKY: wgpu::Color = wgpu::Color { r: 0.62, g: 0.8, b: 1.0, a: 1.0 };

    /// Shared color attachment (sky clear color).
    fn color_attachment<'a>(view: &'a wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'a> {
        wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(Self::SKY),
                store: wgpu::StoreOp::Store,
            },
        }
    }

    /// Shared depth attachment.
    fn depth_attachment<'a>(view: &'a wgpu::TextureView) -> wgpu::RenderPassDepthStencilAttachment<'a> {
        wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }
    }

    /// Present-driven frame (interactive mode).
    fn render_inner(&mut self, meshes: &MeshedChunk, camera: &Camera, tick: u64) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return, // skip frame; window state changed
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let (fw, fh) = (frame.texture.width(), frame.texture.height());
        if (fw, fh) != self.depth_size {
            let depth = Self::make_depth(&self.device, fw, fh);
            self.depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_size = (fw, fh);
        }

        let globals = Globals {
            view_proj: camera.view_proj(),
            anim_offsets: self.animation_offsets(tick),
        };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let color_atts = [Some(Self::color_attachment(&view))];
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                multiview_mask: None,
                color_attachments: &color_atts,
                depth_stencil_attachment: Some(Self::depth_attachment(&self.depth_view)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.record_scene(&mut rpass, meshes, &self.pipelines);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }

    /// Render once into an offscreen texture and copy it to the CPU.
    /// No window surface involvement: no vsync waits, no present, works
    /// before any surface acquisition (deterministic single-frame captures).
    fn render_offscreen(&mut self, meshes: &MeshedChunk, camera: &Camera, tick: u64) {
        let (w, h) = (1280, 720);
        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture-color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // sRGB matches the capture pipeline set (and the window's gamma).
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture-depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let globals = Globals {
            view_proj: camera.view_proj(),
            anim_offsets: self.animation_offsets(tick),
        };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture-readback"),
            size: bytes_per_row as u64 * h as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
        {
            let color_atts = [Some(Self::color_attachment(&color_view))];
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("capture"),
                multiview_mask: None,
                color_attachments: &color_atts,
                depth_stencil_attachment: Some(Self::depth_attachment(&depth_view)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.record_scene(&mut rpass, meshes, &self.capture_pipelines);
        }
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(Ok(())) = rx.recv_timeout(std::time::Duration::from_secs(10)) {
            let Ok(data) = slice.get_mapped_range() else { return };
            let mut out = Vec::with_capacity((w * h * 4) as usize);
            for row in 0..h {
                let start = (row * bytes_per_row) as usize;
                out.extend_from_slice(&data[start..start + (w * 4) as usize]);
            }
            drop(data);
            readback.unmap();
            self.last_frame = Some((w, h, out));
        }
    }


    pub fn format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }
}

fn mip_concat(atlas: &Atlas) -> Vec<u8> {
    let mut out = atlas.pixels.clone();
    for m in &atlas.mips {
        out.extend_from_slice(m);
    }
    out
}

/// Fragment entry for the translucent layer: real alpha blending, no discard.
/// Lives in terrain.wgsl as `fs_main_blend`.

/// Build meshes for a world (used by the client).
pub fn build_meshes(world: &World, registry: &Registry, atlas: &Atlas) -> MeshedChunk {
    let tint = TintPolicy {
        // Plains grass tint; Phase 1 fixed value sampled from the colormap.
        tint0: [145, 189, 89, 255],
    };
    let name_lookup = |sprite_id: u32| -> Option<(u32, u32, u32, u32, u32, u32, bool)> {
        let name = registry.sprite_names().get(sprite_id as usize)?;
        let e = atlas.get(&name.0, &name.1)?;
        // Layer classification: a sprite is "opaque" only if every texel is
        // fully opaque (blocks glass/leaves/plants from the opaque layer).
        let opaque = atlas_opaque(atlas, e);
        Some((e.x, e.y, e.frame_w, e.frame_h, e.frames, e.frame_stride, opaque))
    };
    mesh_world(world, registry, &name_lookup, (atlas.width, atlas.height), &tint)
}

/// Cached per-entry opacity (small set; recomputed per mesh build is fine for
/// Phase 1 — the atlas is ≤2048² and entries are few hundred).
fn atlas_opaque(atlas: &Atlas, e: &feathered_assets::atlas::AtlasEntry) -> bool {
    let stride = atlas.width as usize * 4;
    for row in 0..e.frame_h as usize {
        let y = e.y as usize + row;
        let start = y * stride + e.x as usize * 4;
        let px = &atlas.pixels[start..start + e.frame_w as usize * 4];
        if px.chunks_exact(4).any(|p| p[3] != 255) {
            return false;
        }
    }
    true
}

/// Build the renderer's animation slot table from the registry + atlas.
/// Slot i+1 corresponds to the mesher's `anim_id = i + 1`.
pub fn build_anim_slots(registry: &Registry, atlas: &Atlas) -> Vec<AnimSlot> {
    registry
        .anims()
        .iter()
        .filter_map(|(sid, frametime, frames, strip_frames, interpolate)| {
            let name = registry.sprite_names().get(*sid as usize)?;
            let e = atlas.get(&name.0, &name.1)?;
            Some(AnimSlot {
                frametime: *frametime,
                frames: frames.clone(),
                strip_frames: *strip_frames,
                interpolate: *interpolate,
                stride: e.frame_stride as f32 / atlas.height as f32,
            })
        })
        .collect()
}
