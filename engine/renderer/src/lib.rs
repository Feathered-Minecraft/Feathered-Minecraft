//! feathered-renderer — wgpu pipelines, atlas binding, animation clock.
//!
//! Phase 1 renders three pipelines (opaque, cutout, translucent) sampling the
//! single block atlas. Water animation is driven by a uniform of frame row
//! offsets indexed by `anim_id` (reserved; Phase 1 water uses frame-0 UVs).
//!
//! Phase 2 adds a **modular shader stage graph** configured by
//! [`ShaderEffectConfig`] (see `shader_config.rs`): quality presets map to
//! built-in configurations and external shader packs (via feathered-packs)
//! map their own settings onto the same knobs. The staged pipeline runs
//! gbuffer → (shadow map) → deferred lighting → post. It is *not* any
//! specific shader pack; where algorithms follow published work (including
//! Noble Shaders, GPL-3.0 © Belmu) the provenance is documented in
//! `docs/SHADER_COMPATIBILITY.md` and cited at the use site.
//!
//! API notes (wgpu 30): `create_surface` returns `Result`; acquiring a frame
//! returns the `CurrentSurfaceTexture` enum; presentation goes through
//! `Queue::present(texture)`.

use feathered_assets::atlas::Atlas;
use feathered_chunk::{mesh_world, MeshLayer, MeshedChunk, TintPolicy, Vertex};
use feathered_world::chunks::ChunkPos;
use feathered_world::grid::World;
use feathered_world::{LightGrid, Registry};
use wgpu::util::DeviceExt;

pub mod shader_config;
pub use shader_config::{
    AoConfig, AtmosphereConfig, CloudsConfig, ExposureConfig, FogConfig, PostConfig,
    ShaderEffectConfig, ShadowConfig, Tonemap, WaterConfig,
};

/// Camera: position + yaw/pitch, perspective projection.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

/// Row-major 4×4 operator product `P·V` in WGSL storage order (column j of
/// storage = column j of the mathematical matrix). Shared by the camera and
/// the sun's ortho projection so both shaders multiply identically.
fn compose(proj: [[f32; 4]; 4], view: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
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

fn n3(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Look-at view matrix in the same row-operator convention as
/// `Camera::view_proj` (rows: right/up/−forward + translations).
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let forward = n3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let right = n3(cross3(forward, up));
    let up2 = cross3(right, forward);
    [
        [right[0], right[1], right[2], -dot3(right, eye)],
        [up2[0], up2[1], up2[2], -dot3(up2, eye)],
        [-forward[0], -forward[1], -forward[2], dot3(forward, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

impl Camera {
    /// Unit forward vector (yaw/pitch convention: NEGATIVE pitch = down,
    /// matching the view matrix above).
    pub fn forward(&self) -> [f32; 3] {
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let (sin_p, cos_p) = self.pitch.sin_cos();
        [sin_y * cos_p, sin_p, -cos_y * cos_p]
    }

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

        let view = [
            [right[0], right[1], right[2], -dot3(right, eye)],
            [up[0], up[1], up[2], -dot3(up, eye)],
            [-forward[0], -forward[1], -forward[2], dot3(forward, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Perspective, row-major, [0,1] depth range (Vulkan/wgpu):
        //   z_clip = far/(near-far)·z + far·near/(near-far)
        //   w_clip = -z
        // (b = far·near/(near-far) lives in row 2's homogeneous slot; row 3
        // is the w row.) No y-flip: wgpu's framebuffer rows are top-down, so
        // world-up → NDC-up → lower pixel rows renders upright as-is.
        let f = 1.0 / (self.fov_y / 2.0).tan();
        let (far, near) = (self.far, self.near);
        let proj = [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ];

        compose(proj, view)
    }
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

/// Render quality preset. Mirrors the user-facing preset in feathered-packs
/// (the engine defines its own so the dependency direction stays intact:
/// packs → renderer, never renderer → packs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderQuality {
    /// Half-res scene upscaled, no extras — integrated GPUs / old laptops.
    Low,
    /// Full-res deferred lighting with gentle water; no shadows/atmo/post.
    #[default]
    Medium,
    /// Full-res staged pipeline: shadows, AO, atmosphere, clouds, water,
    /// fog, exposure + tonemap, post bloom + vignette.
    High,
    /// High with heavier per-stage configuration.
    Ultra,
}

impl RenderQuality {
    /// Render-scale factor for the offscreen scene target.
    pub fn render_scale(self) -> f32 {
        match self {
            RenderQuality::Low => 0.5,
            _ => 1.0,
        }
    }

    /// Whether the modular post-process pass runs (legacy path).
    pub fn post_process(self) -> bool {
        matches!(self, RenderQuality::High | RenderQuality::Ultra)
    }

    /// Next preset in Low → Medium → High → Ultra (F4 hotkey).
    pub fn next(self) -> Self {
        match self {
            RenderQuality::Low => RenderQuality::Medium,
            RenderQuality::Medium => RenderQuality::High,
            RenderQuality::High => RenderQuality::Ultra,
            RenderQuality::Ultra => RenderQuality::Low,
        }
    }
}

/// Everything the client can tune about presentation.
#[derive(Debug, Clone, Default)]
pub struct RenderSettings {
    pub quality: RenderQuality,
    /// Active shader pack id, when one is enabled. Informational in the
    /// engine: the pack's *configuration* (validated by feathered-packs)
    /// maps onto the modular `ShaderEffectConfig`; the renderer stays
    /// pack-agnostic.
    pub shader_pack: Option<String>,
}

/// Uniform for the post-process pass (matches post.wgsl `Params`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostParams {
    /// Bit flags: 1 = ACES tonemap, 2 = bloom, 4 = vignette.
    pub flags: [u32; 4],
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub vignette: f32,
    pub _pad: f32,
}

/// Effect flags for the lighting stage (`LightingParams.flags.x`).
pub mod effect_flags {
    pub const SSAO: u32 = 1;
    pub const SHADOWS: u32 = 2;
    pub const ATMOSPHERE: u32 = 4;
    pub const WATER: u32 = 8;
    pub const FOG: u32 = 16;
}

/// Lighting-stage uniform (matches lighting.wgsl `Params`). All vec4-aligned.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingParams {
    pub flags: [u32; 4],
    /// xyz camera (blocks), w time in seconds.
    pub camera: [f32; 4],
    /// xyz sun direction (normalized), w sun elevation sin.
    pub sun_dir: [f32; 4],
    /// x mie g, y sun intensity, z ambient strength, w unused.
    pub atmosphere: [f32; 4],
    /// rgb rayleigh β, w mie β.
    pub rayleigh: [f32; 4],
    /// x water octaves, y amplitude, z speed, w reflection strength.
    pub water: [f32; 4],
    /// rgb water absorption, w fog density.
    pub absorption: [f32; 4],
    /// rgb fog tint, w exposure scale.
    pub fog_tint: [f32; 4],
    /// x ao strength, y ao radius, z near, w far.
    pub ao: [f32; 4],
    /// x altitude, y thickness, z coverage, w steps (0 = clouds off).
    pub clouds: [f32; 4],
    /// x density, y wind speed, z/w unused.
    pub clouds_b: [f32; 4],
    /// Normalized world-space ray directions to the frustum corners
    /// (top-left, top-right, bottom-left, bottom-right) — sky pixels
    /// reconstruct their view ray by bilinear interpolation (exact for
    /// perspective frustums).
    pub ray_tl: [f32; 4],
    pub ray_tr: [f32; 4],
    pub ray_bl: [f32; 4],
    pub ray_br: [f32; 4],
    pub inv_view_proj: [[f32; 4]; 4],
}

/// Shadow-stage uniform (matches lighting.wgsl `ShadowParams`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowParams {
    pub sun_view_proj: [[f32; 4]; 4],
    /// x map extent (± blocks), y depth bias, z strength, w unused.
    pub config: [f32; 4],
}

/// CPU-authored overlay geometry for one frame (see client/src/overlay.rs).
/// The client fills the tri-lists; the renderer uploads and draws them on
/// top of the finished frame.
#[derive(Default)]
pub struct HudDraw {
    /// Screen-space triangles (UI pixels): crosshair, hotbar, text, progress.
    pub screen: client_overlay::TriList,
    /// World-space triangles (blocks): block outline, particles.
    pub world: client_overlay::TriList,
}

/// Client-side overlay vertex/list types, re-exported for the renderer's
/// public API without a crate dependency cycle (the types are POD mirrors).
pub mod client_overlay {
    /// Mirrors feathered_client::overlay::HudVertex (32 bytes: 12 pos + 4
    /// color + 8 UV-in-px + 8 reserved, no padding).
    #[repr(C)]
    #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct HudVertex {
        pub pos: [f32; 3],
        pub color: [u8; 4],
        /// Screen pass: UV into the overlay texture (x < 0 = flat quad).
        pub px: [f32; 2],
        /// Reserved (layout padding shared with the world-space pass).
        pub uv: [f32; 2],
    }
    const _: () = assert!(std::mem::size_of::<HudVertex>() == 32);

    /// Mirrors feathered_client::overlay::TriList.
    #[derive(Debug, Default)]
    pub struct TriList {
        pub vertices: Vec<HudVertex>,
        pub indices: Vec<u32>,
    }
}

/// Persistent GPU state for the overlay system (HUD + outlines). Buffers
/// are recreated (not resized) per frame only when the draw list grows —
/// normal HUD churn is a few hundred quads.
struct OverlayState {
    screen_params_buf: wgpu::Buffer,
    screen_bind: wgpu::BindGroup,
    screen_pipeline: wgpu::RenderPipeline,
    screen_pipeline_capture: wgpu::RenderPipeline,
    /// Overlay art texture (menu logo); None = 1×1 transparent fallback and
    /// a bind group pointing at it (textured quads then draw nothing).
    logo_tex: Option<wgpu::Texture>,
    logo_view: wgpu::TextureView,
    logo_sampler: wgpu::Sampler,
    logo_bind: wgpu::BindGroup,
    /// Full-screen menu background art (second texture slot, bindings 3/4).
    bg_tex: Option<wgpu::Texture>,
    bg_view: wgpu::TextureView,
    bg_sampler: wgpu::Sampler,
    /// Texture bind group layout (reused when the logo is re-uploaded).
    tex_bgl: wgpu::BindGroupLayout,
    /// Outline pass binds the shared globals (view_proj) + scene depth.
    outline_pipeline: wgpu::RenderPipeline,
    outline_pipeline_capture: wgpu::RenderPipeline,
    screen_vbuf: wgpu::Buffer,
    screen_ibuf: wgpu::Buffer,
    /// Buffer capacities (vertices / indices) — grown lazily.
    screen_vcap: usize,
    screen_icap: usize,
    /// Indices to draw this frame (0 = skip the pass).
    screen_count: u32,
    outline_vbuf: wgpu::Buffer,
    outline_ibuf: wgpu::Buffer,
    outline_vcap: usize,
    outline_icap: usize,
    outline_count: u32,
}

/// One chunk's persistent GPU geometry.
pub struct GpuChunk {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
}

/// Upload a meshed chunk into persistent GPU buffers. The renderer keeps
/// these for every frame (chunk streaming re-uploads only changed chunks);
/// the legacy per-frame `create_buffer_init` path stays for the Phase-1
/// single-mesh API.
pub fn upload_meshed(device: &wgpu::Device, mesh: &MeshedChunk) -> [Option<GpuChunk>; 3] {
    let mk = |layer: &MeshLayer| -> Option<GpuChunk> {
        if layer.indices.is_empty() {
            return None;
        }
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-vbuf"),
            contents: bytemuck::cast_slice(&layer.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-ibuf"),
            contents: bytemuck::cast_slice(&layer.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some(GpuChunk {
            vertex_buf,
            index_buf,
            index_count: layer.indices.len() as u32,
        })
    };
    [mk(&mesh.opaque), mk(&mesh.cutout), mk(&mesh.translucent)]
}

impl OverlayState {
    const OVERLAY_VBUF_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: 32,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Unorm8x4,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 24,
                shader_location: 3,
            },
        ],
    };

    const INITIAL_VERTS: usize = 4096;
    const INITIAL_INDICES: usize = 6144;

    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        capture_format: wgpu::TextureFormat,
        globals_bgl: &wgpu::BindGroupLayout,
        globals_bind: &wgpu::BindGroup,
    ) -> OverlayState {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });

        let screen_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay-screen-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let screen_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay-screen-params"),
            contents: bytemuck::bytes_of(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let screen_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay-screen-bg"),
            layout: &screen_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_params_buf.as_entire_binding(),
            }],
        });

        // NOTE: the outline layout also carries the texture BGL — the shared
        // fragment entry point statically references the texture even though
        // outline vertices pass the flat-UV sentinel (naga requires the
        // binding in the layout regardless of runtime branching).
        let tex_bgl_early = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay-tex-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let outline_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay-outline-pl"),
            bind_group_layouts: &[Some(globals_bgl), Some(&tex_bgl_early)],
            immediate_size: 0,
        });

        let mk_pipeline = |label: &'static str,
                           layout: &wgpu::PipelineLayout,
                           entry: &'static str,
                           fmt: wgpu::TextureFormat,
                           depth: Option<wgpu::DepthStencilState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(entry),
                    buffers: &[Some(Self::OVERLAY_VBUF_LAYOUT)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: depth,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let with_depth = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            // Outlines hug the block faces: LessEqual + a tiny bias keeps
            // the expanded beam visible without shadow-acne flicker.
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: -1,
                slope_scale: -1.0,
                clamp: 0.0,
            },
        });

        // Screen pipelines are built below with the texture bind group (the
        // WGSL fragment declares group 0 binding 1/2 unconditionally).
        let outline_pipeline = mk_pipeline(
            "overlay-outline",
            &outline_pl,
            "vs_outline",
            surface_format,
            with_depth.clone(),
        );
        let outline_pipeline_capture = mk_pipeline(
            "overlay-outline-capture",
            &outline_pl,
            "vs_outline",
            capture_format,
            with_depth,
        );
        let _ = globals_bind; // bound per-frame (shared); kept for clarity

        // Separate buffer pairs per pass: both are written every frame.
        // The vertex capacity is in HudVertex units (32 B — the same stride
        // grow_and_upload uses; it was 24 here, undersizing the initial
        // buffers by 25% so a full title-screen draw list overran them).
        let hud_vert_bytes = std::mem::size_of::<client_overlay::HudVertex>() as u64;
        debug_assert_eq!(hud_vert_bytes, 32);
        let mk_bufs = |label: &'static str| {
            (
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: Self::INITIAL_VERTS as u64 * hud_vert_bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (Self::INITIAL_INDICES * 4) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            )
        };
        let (screen_vbuf, screen_ibuf) = mk_bufs("overlay-screen-buf");
        let (outline_vbuf, outline_ibuf) = mk_bufs("overlay-outline-buf");

        // Overlay art texture: 1×1 transparent until `set_logo_texture`
        // replaces it (flat quads are unaffected; textured quads then tint
        // real texels). Linear sampling so downscaled art stays smooth.
        let logo_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay-logo-fallback"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let logo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay-logo-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay-tex-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Background fallback: 1×1 transparent (menu_background.png
        // replaces it; until then bg quads draw nothing).
        let (bg_view, bg_sampler) = (logo_view.clone(), logo_sampler.clone());
        let logo_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay-tex-bg"),
            layout: &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&logo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&logo_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&bg_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&bg_sampler),
                },
            ],
        });
        // Rebuild the screen pipelines: their layout now includes the texture
        // bind group (module is recompiled from the same WGSL source).
        let screen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay-screen-tex"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("overlay-screen-tex-pl"),
                    bind_group_layouts: &[Some(&screen_params_bgl), Some(&tex_bgl)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_screen"),
                buffers: &[Some(Self::OVERLAY_VBUF_LAYOUT)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let screen_pipeline_capture =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay-screen-tex-capture"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("overlay-screen-tex-capture-pl"),
                        bind_group_layouts: &[Some(&screen_params_bgl), Some(&tex_bgl)],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_screen"),
                    buffers: &[Some(Self::OVERLAY_VBUF_LAYOUT)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: capture_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        OverlayState {
            screen_params_buf,
            screen_bind,
            screen_pipeline,
            screen_pipeline_capture,
            logo_tex: None,
            logo_view,
            logo_sampler,
            logo_bind,
            bg_tex: None,
            bg_view,
            bg_sampler,
            tex_bgl,
            outline_pipeline,
            outline_pipeline_capture,
            screen_vbuf,
            screen_ibuf,
            screen_vcap: Self::INITIAL_VERTS,
            screen_icap: Self::INITIAL_INDICES,
            screen_count: 0,
            outline_vbuf,
            outline_ibuf,
            outline_vcap: Self::INITIAL_VERTS,
            outline_icap: Self::INITIAL_INDICES,
            outline_count: 0,
        }
    }

    /// Rebuild the shared texture bind group after a background upload
    /// (bindings 3/4 swap to the new view; logo slots are re-bound as-is).
    fn bg_bind_rebuild(&mut self, device: &wgpu::Device, view: &wgpu::TextureView) {
        self.logo_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay-tex-bg"),
            layout: &self.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.logo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.logo_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.bg_sampler),
                },
            ],
        });
    }

    /// Grow a buffer pair when a draw list outgrew it, then upload the
    /// list. Caps are updated in place (capacity semantics).
    fn grow_and_upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        list: &client_overlay::TriList,
        vbuf: &mut wgpu::Buffer,
        ibuf: &mut wgpu::Buffer,
        cap_v: &mut usize,
        cap_i: &mut usize,
    ) {
        if list.vertices.len() > *cap_v || list.indices.len() > *cap_i {
            let nv = (list.vertices.len() * 2).max(Self::INITIAL_VERTS);
            let ni = (list.indices.len() * 2).max(Self::INITIAL_INDICES);
            *vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay-vbuf"),
                size: (nv * 32) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *ibuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay-ibuf"),
                size: (ni * 4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cap_v = nv;
            *cap_i = ni;
        }
        if !list.vertices.is_empty() {
            queue.write_buffer(vbuf, 0, bytemuck::cast_slice(&list.vertices));
            queue.write_buffer(ibuf, 0, bytemuck::cast_slice(&list.indices));
        }
    }

    /// Upload the pending draw lists and record how much to draw.
    fn upload_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &client_overlay::TriList,
        outline: &client_overlay::TriList,
    ) {
        let Self {
            screen_vbuf,
            screen_ibuf,
            screen_vcap,
            screen_icap,
            screen_count,
            outline_vbuf,
            outline_ibuf,
            outline_vcap,
            outline_icap,
            outline_count,
            ..
        } = self;
        Self::grow_and_upload(
            device,
            queue,
            screen,
            screen_vbuf,
            screen_ibuf,
            screen_vcap,
            screen_icap,
        );
        *screen_count = screen.indices.len() as u32;
        Self::grow_and_upload(
            device,
            queue,
            outline,
            outline_vbuf,
            outline_ibuf,
            outline_vcap,
            outline_icap,
        );
        *outline_count = outline.indices.len() as u32;
    }
}

/// Sun direction for a fraction of the day. The path is a great circle
/// through the zenith tilted by `sun_path_rotation_deg` (Noble's
/// `sunPathRotation` convention: rotation of the celestial plane).
/// Returns (normalized direction, sin of elevation).
pub fn sun_state(sun_angle: f32, rotation_deg: f32) -> ([f32; 3], f32) {
    let theta = sun_angle * std::f32::consts::TAU; // 0 = sunrise, 0.25 = noon
    let rot = rotation_deg.to_radians();
    let d = [
        theta.cos(),
        theta.sin() * rot.cos(),
        theta.sin() * rot.sin(),
    ];
    let d = n3(d);
    (d, d[1])
}

/// The renderer. Owns device, pipelines and GPU resources.
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// `None` for headless renderers (tests / screenshot batches).
    surface: Option<wgpu::Surface<'static>>,
    surface_config: wgpu::SurfaceConfiguration,
    pipelines: LayerPipelines,
    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    /// Kept for late pipeline construction (overlay outline pass binds the
    /// same globals layout).
    globals_bgl: wgpu::BindGroupLayout,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
    anims: Vec<AnimSlot>,
    /// Pipeline set targeting Rgba8UnormSrgb (offscreen captures AND the
    /// legacy scene path).
    capture_pipelines: LayerPipelines,
    /// CPU copy of the last captured frame. Not allocated until first capture.
    last_frame: Option<(u32, u32, Vec<u8>)>,
    /// Presentation settings (quality preset, active shader pack).
    settings: RenderSettings,
    /// Post-process pipeline for the window's surface format.
    post_window: Option<wgpu::RenderPipeline>,
    /// Post-process pipeline for the Rgba8UnormSrgb capture target.
    post_capture: Option<wgpu::RenderPipeline>,
    post_params_buf: wgpu::Buffer,
    post_bind_layout: wgpu::BindGroupLayout,
    post_sampler: wgpu::Sampler,
    /// Offscreen scene targets (gbuffer + lit + depth).
    scene: Option<SceneTargets>,
    post_bind_group: Option<wgpu::BindGroup>,
    /// Modular shader-stage state. `None` when the effective configuration
    /// needs no staging (Low preset, and packs that disable every stage).
    staged: Option<StagedState>,
    /// World lighting + relit mesh for the deferred stage's lightmap input.
    world_light: Option<LightGrid>,
    mesh_lighting: Option<MeshedChunk>,
    /// Streaming chunks: persistent GPU buffers keyed by chunk position.
    /// Drawn every frame in addition to (or instead of) the legacy mesh.
    chunks: std::collections::HashMap<ChunkPos, [Option<GpuChunk>; 3]>,
    /// Day fraction (0..1) the client drives for the day/night cycle. `None`
    /// keeps the configured/pack sun angle (Phase-2 behavior).
    day_fraction: Option<f32>,
    /// Absolute sun phase override (0..1). When set it wins over both the
    /// client's day cycle and the shader config's sun angle — the title
    /// screen pins its golden-hour backdrop with this. Cleared when a
    /// world starts so gameplay follows the day cycle again.
    sun_phase_override: Option<f32>,
    /// Server-tick counter from the client (drives animation + day cycle).
    tick: u64,
    /// HUD + overlay GPU state (None until the first `set_overlay`).
    overlay: Option<OverlayState>,
    /// Pending overlay geometry for the next frame (set by `set_overlay`).
    pending_overlay: Option<HudDraw>,
    /// Logo set before overlay init (rare: one-frame delay). Flushed when
    /// the overlay state is created in `set_overlay`.
    pending_logo: Option<(u32, u32, Vec<u8>)>,
    /// Background set before overlay init (flushed in set_overlay).
    pending_bg: Option<(u32, u32, Vec<u8>)>,
}

/// Offscreen targets for the scene path (both legacy and staged share these;
/// the staged path uses lightmap + lit, the legacy path only albedo+depth).
struct SceneTargets {
    /// Albedo + face shade (Rgba8UnormSrgb); legacy post input.
    /// The textures are never read directly — every consumer goes through
    /// the views below — but they own the GPU memory, so dropping them
    /// would invalidate the views.
    #[allow(dead_code)]
    albedo: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    /// (sky, block) light levels; alpha doubles as the water marker.
    #[allow(dead_code)]
    lightmap: wgpu::Texture,
    lightmap_view: wgpu::TextureView,
    /// Lighting-stage output (post pass input when staging), Rgba8UnormSrgb.
    #[allow(dead_code)]
    lit: wgpu::Texture,
    lit_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    w: u32,
    h: u32,
}

/// Everything the staged pipeline allocates per configuration.
struct StagedState {
    config: ShaderEffectConfig,
    gbuffer: GbufferPipelines,
    shadow_pipeline: wgpu::RenderPipeline,
    lighting_pipeline: wgpu::RenderPipeline,
    lighting_params_buf: wgpu::Buffer,
    shadow_params_buf: wgpu::Buffer,
    lighting_bind_layout: wgpu::BindGroupLayout,
    lighting_bind: Option<wgpu::BindGroup>,
    /// Second globals buffer + bind group holding the SUN's ortho matrix for
    /// the shadow pass (the camera matrix stays in `globals_buf`).
    sun_globals_buf: wgpu::Buffer,
    sun_globals_bind: wgpu::BindGroup,
    shadow: ShadowMap,
    /// Depth-comparison sampler for the shadow map.
    shadow_sampler: wgpu::Sampler,
}

/// Gbuffer pipeline set (two color targets: albedo+shade, lightmap).
struct GbufferPipelines {
    opaque: wgpu::RenderPipeline,
    cutout: wgpu::RenderPipeline,
    translucent: wgpu::RenderPipeline,
    atlas_bind: wgpu::BindGroup,
}

struct ShadowMap {
    /// Kept alive for `view` (same ownership rule as SceneTargets).
    #[allow(dead_code)]
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    #[allow(dead_code)]
    size: u32,
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
        settings: RenderSettings,
    ) -> Renderer {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window).expect("create surface");
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
        Self::construct_tail(
            device,
            queue,
            Some(surface),
            surface_config,
            atlas,
            anims,
            settings,
            width,
            height,
        )
    }

    /// Shared construction tail: atlas upload, pipelines, post pass. Used by
    /// both the windowed and headless constructors.
    #[allow(clippy::too_many_arguments)]
    fn construct_tail(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: Option<wgpu::Surface<'static>>,
        surface_config: wgpu::SurfaceConfiguration,
        atlas: &Atlas,
        anims: Vec<AnimSlot>,
        settings: RenderSettings,
        width: u32,
        height: u32,
    ) -> Renderer {
        if let Some(s) = &surface {
            s.configure(&device, &surface_config);
        }

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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm16x2,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint8x2,
                    offset: 20,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x2,
                    offset: 22,
                    shader_location: 4,
                },
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
                        format: wgpu::TextureFormat::Depth32Float,
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

        let surface_format = surface_config.format;
        let pipelines = make_pipelines(surface_format);
        let capture_pipelines = make_pipelines(wgpu::TextureFormat::Rgba8UnormSrgb);

        // Post-process pipeline (modular pass). Built eagerly so the staged
        // path can always composite (vignette/upscale) through it.
        let post_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
        });
        let post_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let post_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-pl"),
            bind_group_layouts: &[Some(&post_bind_layout)],
            immediate_size: 0,
        });
        let make_post = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post"),
                layout: Some(&post_pl),
                vertex: wgpu::VertexState {
                    module: &post_module,
                    entry_point: Some("vs_post"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_module,
                    entry_point: Some("fs_post"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let post_window = make_post(surface_format);
        let post_capture = make_post(wgpu::TextureFormat::Rgba8UnormSrgb);
        let post_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post-params"),
            contents: bytemuck::bytes_of(&PostParams {
                flags: [0; 4],
                bloom_threshold: 0.0,
                bloom_intensity: 0.0,
                vignette: 0.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let post_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Renderer {
            pending_logo: None,
            pending_bg: None,
            device,
            queue,
            surface,
            surface_config,
            pipelines,
            globals_buf,
            globals_bind,
            globals_bgl,
            depth_view,
            depth_size: (width.max(1), height.max(1)),
            anims,
            capture_pipelines,
            last_frame: None,
            post_window: Some(post_window),
            post_capture: Some(post_capture),
            post_params_buf,
            post_bind_layout,
            post_sampler,
            settings,
            scene: None,
            post_bind_group: None,
            staged: None,
            world_light: None,
            mesh_lighting: None,
            chunks: Default::default(),
            day_fraction: None,
            sun_phase_override: None,
            tick: 0,
            overlay: None,
            pending_overlay: None,
        }
        .with_staged()
    }

    /// Build the staged pipeline set after `new` (builder-style tail).
    fn with_staged(mut self) -> Renderer {
        self.rebuild_staged();
        self
    }

    /// Headless renderer for tests and batch screenshot runs: same pipelines,
    /// no window surface. Only `render_capture` works (no present path).
    pub async fn headless(
        width: u32,
        height: u32,
        atlas: &Atlas,
        anims: Vec<AnimSlot>,
        settings: RenderSettings,
    ) -> Renderer {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("no suitable adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("feathered-headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        Self::construct_tail(
            device,
            queue,
            None,
            surface_config,
            atlas,
            anims,
            settings,
            width,
            height,
        )
    }

    fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
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
        if let Some(s) = &self.surface {
            s.configure(&self.device, &self.surface_config);
        }
        let depth = Self::make_depth(&self.device, width, height);
        self.depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_size = (width, height);
    }

    /// Render one frame (window mode only; headless renderers capture).
    pub fn render(&mut self, meshes: &MeshedChunk, camera: &Camera, tick: u64) {
        self.tick = tick;
        let Some(surface) = &self.surface else { return };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return, // skip frame; window state changed
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
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
        self.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        self.queue.write_buffer(
            &self.post_params_buf,
            0,
            bytemuck::bytes_of(&self.post_params()),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.record_frame(&mut encoder, meshes, camera, tick, &view, true);
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }

    /// Render one frame into an offscreen target and keep a CPU copy for
    /// `capture_last_frame` (no present, no vsync — deterministic captures).
    pub fn render_capture(&mut self, meshes: &MeshedChunk, camera: &Camera, tick: u64) {
        let (w, h) = (1280u32, 720u32);
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture-readback"),
            size: bytes_per_row as u64 * h as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture-color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());

        let globals = Globals {
            view_proj: camera.view_proj(),
            anim_offsets: self.animation_offsets(tick),
        };
        self.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        self.queue.write_buffer(
            &self.post_params_buf,
            0,
            bytemuck::bytes_of(&self.post_params()),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture"),
            });
        self.record_frame(&mut encoder, meshes, camera, tick, &color_view, false);

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
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(Ok(())) = rx.recv_timeout(std::time::Duration::from_secs(10)) {
            let Ok(data) = slice.get_mapped_range() else {
                return;
            };
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

    /// Record one full frame into `view` (window or capture target).
    /// Chooses: staged pipeline → legacy scaled scene path → direct path.
    fn record_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        meshes: &MeshedChunk,
        camera: &Camera,
        tick: u64,
        view: &wgpu::TextureView,
        to_window: bool,
    ) {
        if self.staged.is_some() {
            let (fw, fh) = self.frame_dims(to_window);
            let (sw, sh) = self.scene_size_fw_fh(fw, fh);
            self.ensure_scene(sw.max(64), sh.max(64));
            self.record_staged(encoder, meshes, camera, tick);
            // Composite lit → final target through the post pass (bloom +
            // vignette; tonemap already happened in the lighting stage).
            let post = if to_window {
                self.post_window.clone()
            } else {
                self.post_capture.clone()
            };
            if let Some(post) = post {
                self.rebind_post_to_lit();
                self.record_post_pass(encoder, view, &post);
            }
            // Overlays: outlines depth-test against the staged scene depth;
            // the HUD draws over the composited frame.
            let scene_depth = self.scene.as_ref().map(|s| s.depth_view.clone());
            self.record_overlays(encoder, view, scene_depth.as_ref(), to_window);
            return;
        }

        if self.settings.quality.post_process() || self.settings.quality.render_scale() < 1.0 {
            let (fw, fh) = self.frame_dims(to_window);
            let (sw, sh) = self.scene_size_fw_fh(fw, fh);
            self.ensure_scene(sw.max(64), sh.max(64));
            self.ensure_post_bind();
            {
                let scene = self.scene.as_ref().unwrap();
                let color_atts = [Some(Self::color_attachment(&scene.albedo_view))];
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main-scene"),
                    multiview_mask: None,
                    color_attachments: &color_atts,
                    depth_stencil_attachment: Some(Self::depth_attachment(&scene.depth_view)),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                self.record_scene(&mut rpass, meshes, &self.capture_pipelines);
            }
            if let Some(post) = if to_window {
                self.post_window.clone()
            } else {
                self.post_capture.clone()
            } {
                self.record_post_pass(encoder, view, &post);
            }
            let scene_depth = self.scene.as_ref().map(|s| s.depth_view.clone());
            self.record_overlays(encoder, view, scene_depth.as_ref(), to_window);
            return;
        }

        // Direct path: straight into the target. The scene draw and the
        // outline overlay share one depth buffer: the capture outline
        // pipeline declares a Depth32Float attachment, so a queued outline
        // with no depth view would make that render pass invalid (wgpu
        // validation error). Window targets reuse the persistent depth.
        let depth_view = if to_window {
            self.depth_view.clone()
        } else {
            // Capture target without staging: give it its own depth.
            Self::make_depth(&self.device, 1280, 720)
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        {
            let color_atts = [Some(Self::color_attachment(view))];
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                multiview_mask: None,
                color_attachments: &color_atts,
                depth_stencil_attachment: Some(Self::depth_attachment(&depth_view)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let pipes = if to_window {
                &self.pipelines
            } else {
                &self.capture_pipelines
            };
            self.record_scene(&mut rpass, meshes, pipes);
        }
        self.record_overlays(encoder, view, Some(&depth_view), to_window);
    }

    /// Frame dimensions for window vs capture targets.
    fn frame_dims(&self, to_window: bool) -> (u32, u32) {
        if to_window && self.surface.is_some() {
            (self.surface_config.width, self.surface_config.height)
        } else {
            (1280, 720) // capture targets are fixed-size like render_offscreen
        }
    }

    fn scene_size_fw_fh(&self, w: u32, h: u32) -> (u32, u32) {
        let s = self.settings.quality.render_scale();
        (
            ((w as f32 * s).round() as u32).max(1),
            ((h as f32 * s).round() as u32).max(1),
        )
    }

    /// Record the staged pipeline: shadow map → gbuffer → deferred lighting
    /// (lit output lands in the scene's `lit` target).
    fn record_staged(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        meshes: &MeshedChunk,
        camera: &Camera,
        tick: u64,
    ) {
        let Some(staged) = &self.staged else { return };
        let config = staged.config.clone();

        // --- Lighting uniforms --------------------------------------------
        // Day/night cycle: the client supplies a day fraction (0..1); the
        // configured sun_angle becomes the phase within it. Without a cycle
        // (tests, captures) the configured angle is used directly.
        // Title-backdrop override first: an absolute phase wins over both
        // the day cycle and the configured/pack sun angle.
        let sun_angle = match self.sun_phase_override {
            Some(p) => p,
            None => match self.day_fraction {
                Some(day) => (day + config.sun_angle).fract(),
                None => config.sun_angle,
            },
        };
        let (sun, elev) = sun_state(sun_angle, config.sun_path_rotation_deg);
        let mut flags = 0u32;
        if config.ssao.is_some() {
            flags |= effect_flags::SSAO;
        }
        if config.shadows.is_some() {
            flags |= effect_flags::SHADOWS;
        }
        if config.atmosphere.is_some() {
            flags |= effect_flags::ATMOSPHERE;
        }
        if config.water.is_some() {
            flags |= effect_flags::WATER;
        }
        if config.fog.is_some() {
            flags |= effect_flags::FOG;
        }

        let lparams = LightingParams {
            flags: [
                flags,
                config
                    .shadows
                    .as_ref()
                    .map(|s| s.samples.min(16))
                    .unwrap_or(0),
                match config
                    .post
                    .as_ref()
                    .map(|p| p.tonemap)
                    .unwrap_or(Tonemap::None)
                {
                    Tonemap::None => 0,
                    Tonemap::Aces => 1,
                    Tonemap::Reinhard => 2,
                },
                0,
            ],
            camera: [
                camera.pos[0],
                camera.pos[1],
                camera.pos[2],
                tick as f32 / 20.0,
            ],
            // daylight rides in atmosphere.w (ambient strength): dusk/night
            // dims ambient so the world darkens smoothly.
            sun_dir: [sun[0], sun[1], sun[2], elev],
            atmosphere: [
                config.atmosphere.as_ref().map(|a| a.mie_g).unwrap_or(0.76),
                config
                    .atmosphere
                    .as_ref()
                    .map(|a| a.sun_illuminance)
                    .unwrap_or(30.0),
                ambient_for_sun(elev),
                0.0,
            ],
            rayleigh: config
                .atmosphere
                .as_ref()
                .map(|a| [a.rayleigh[0], a.rayleigh[1], a.rayleigh[2], a.mie])
                .unwrap_or([5.8e-6, 13.5e-6, 33.1e-6, 21e-6]),
            water: config
                .water
                .as_ref()
                .map(|w| {
                    [
                        w.octaves as f32,
                        w.amplitude,
                        w.speed,
                        w.reflection_strength,
                    ]
                })
                .unwrap_or([4.0, 0.5, 1.0, 0.6]),
            absorption: config
                .water
                .as_ref()
                .map(|w| [w.absorption[0], w.absorption[1], w.absorption[2], 0.0])
                .unwrap_or([0.45, 0.08, 0.03, 0.0]),
            fog_tint: {
                let (t, _) = config
                    .fog
                    .as_ref()
                    .map(|f| (f.tint, f.density))
                    .unwrap_or(([1.0, 1.0, 1.0], 0.0));
                [t[0], t[1], t[2], 0.0]
            },
            ao: [
                config.ssao.as_ref().map(|a| a.strength).unwrap_or(0.0),
                config.ssao.as_ref().map(|a| a.radius).unwrap_or(3.0),
                camera.near,
                camera.far,
            ],
            clouds: config
                .clouds
                .as_ref()
                .map(|c| [c.altitude, c.thickness, c.coverage, c.steps as f32])
                .unwrap_or([0.0, 0.0, 0.0, 0.0]),
            clouds_b: config
                .clouds
                .as_ref()
                .map(|c| [c.density, c.wind_speed, 0.0, 0.0])
                .unwrap_or([0.0, 0.0, 0.0, 0.0]),
            ray_tl: corner_ray(camera, -1.0, 1.0),
            ray_tr: corner_ray(camera, 1.0, 1.0),
            ray_bl: corner_ray(camera, -1.0, -1.0),
            ray_br: corner_ray(camera, 1.0, -1.0),
            inv_view_proj: inverse_view_proj(camera),
        };
        // Fog density rides in absorption.w; exposure scale in fog_tint.w.
        //
        // Exposure calibration: Noble's EV100 formula is absolute (its scene
        // units put the sun at ~1e5 luminance), so `2^-EV100` alone would
        // crush Feathered's HDR range (sun_illuminance ≈ 40) to black. The
        // formula itself stays Noble-exact — see ExposureConfig::exposure_scale
        // — and the application site calibrates it to Feathered's units with
        // EV_REF (f/8, 1/125s, ISO 200 → scale 1.0). Pack translations of
        // F_STOPS/ISO/SHUTTER_SPEED therefore shift exposure *relative* to
        // that reference exactly as Noble's knobs do relative to his scene.
        const EV_REF_SCALE: f32 = 4000.0; // 2^11.97
        let mut lparams = lparams;
        lparams.absorption[3] = config.fog.as_ref().map(|f| f.density).unwrap_or(0.0);
        lparams.fog_tint[3] = config
            .exposure
            .as_ref()
            .map(|e| (e.exposure_scale() * EV_REF_SCALE).clamp(0.05, 8.0))
            .unwrap_or(1.0);
        if std::env::var("FEATHERED_MARK_ATMO").is_ok() {
            lparams.flags[3] = 777; // raw atmosphere debug view
        }
        if std::env::var("FEATHERED_MARK_UNIFORM").is_ok() {
            lparams.flags[3] = 778; // uniform echo debug view
        }
        if std::env::var("FEATHERED_MARK_SHADOW").is_ok() {
            lparams.flags[3] = 779; // raw pcf_visibility debug view
        }
        if std::env::var("FEATHERED_MARK_SHADOWMAP").is_ok() {
            lparams.flags[3] = 780; // shadow-projection footprint debug view
        }
        let (params_buf, shadow_buf) = {
            let s = self.staged.as_ref().unwrap();
            (s.lighting_params_buf.clone(), s.shadow_params_buf.clone())
        };
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&lparams));

        // --- Shadow map pass ----------------------------------------------
        if let Some(sh) = &config.shadows {
            let (sun_vp, extent) = sun_view_proj(camera.pos, &sun, sh.distance);
            let sparams = ShadowParams {
                sun_view_proj: sun_vp,
                config: [extent, 0.0022, sh.strength, 0.0],
            };
            self.queue
                .write_buffer(&shadow_buf, 0, bytemuck::bytes_of(&sparams));
            // vs_shadow transforms through globals.view_proj — bind the SUN's
            // ortho matrix for this pass (animation offsets irrelevant: the
            // shadow fragment only tests alpha coverage).
            let sun_globals_buf = self.staged.as_ref().unwrap().sun_globals_buf.clone();
            self.queue.write_buffer(
                &sun_globals_buf,
                0,
                bytemuck::bytes_of(&Globals {
                    view_proj: sun_vp,
                    anim_offsets: [0.0; 64],
                }),
            );

            let sun_bind = self.staged.as_ref().unwrap().sun_globals_bind.clone();
            let shadow_view = self.staged.as_ref().unwrap().shadow.view.clone();
            let shadow_pipe = self.staged.as_ref().unwrap().shadow_pipeline.clone();
            {
                let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow-map"),
                    multiview_mask: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &shadow_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let mesh = self.mesh_lighting.as_ref().unwrap_or(meshes);
                spass.set_bind_group(0, &sun_bind, &[]);
                let gbuffer = &self.staged.as_ref().unwrap().gbuffer;
                // Streaming chunks cast shadows too (persistent buffers).
                for gpu in self.chunks.values() {
                    for layer in [&gpu[0], &gpu[1]] {
                        let Some(g) = layer else { continue };
                        spass.set_pipeline(&shadow_pipe);
                        spass.set_bind_group(1, &gbuffer.atlas_bind, &[]);
                        spass.set_vertex_buffer(0, g.vertex_buf.slice(..));
                        spass.set_index_buffer(g.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                        spass.draw_indexed(0..g.index_count, 0, 0..1);
                    }
                }
                for layer in [&mesh.opaque, &mesh.cutout] {
                    // Water does not cast (fs_shadow discards it anyway).
                    if layer.indices.is_empty() {
                        continue;
                    }
                    let vbuf = self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::cast_slice(&layer.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let ibuf = self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::cast_slice(&layer.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                    spass.set_pipeline(&shadow_pipe);
                    spass.set_bind_group(1, &gbuffer.atlas_bind, &[]);
                    spass.set_vertex_buffer(0, vbuf.slice(..));
                    spass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    spass.draw_indexed(0..layer.indices.len() as u32, 0, 0..1);
                }
            }
        }

        // --- Gbuffer pass ---------------------------------------------------
        {
            let (albedo_view, lightmap_view, depth_view) = {
                let s = self.scene.as_ref().unwrap();
                (
                    s.albedo_view.clone(),
                    s.lightmap_view.clone(),
                    s.depth_view.clone(),
                )
            };
            let color_atts = [
                Some(Self::color_attachment_clear(&albedo_view, Self::SKY)),
                Some(Self::color_attachment_clear(
                    &lightmap_view,
                    wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    },
                )),
            ];
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer"),
                multiview_mask: None,
                color_attachments: &color_atts,
                depth_stencil_attachment: Some(Self::depth_attachment(&depth_view)),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mesh = self.mesh_lighting.as_ref().unwrap_or(meshes);
            rpass.set_bind_group(0, &self.globals_bind, &[]);
            let gbuffer = &self.staged.as_ref().unwrap().gbuffer;
            // Streaming chunks (persistent buffers) into the gbuffer.
            {
                let gbuffer = &self.staged.as_ref().unwrap().gbuffer;
                for (pipe, slot) in [
                    (&gbuffer.opaque, 0usize),
                    (&gbuffer.cutout, 1),
                    (&gbuffer.translucent, 2),
                ] {
                    for gpu in self.chunks.values() {
                        let Some(g) = &gpu[slot] else { continue };
                        rpass.set_pipeline(pipe);
                        rpass.set_bind_group(1, &gbuffer.atlas_bind, &[]);
                        rpass.set_vertex_buffer(0, g.vertex_buf.slice(..));
                        rpass.set_index_buffer(g.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                        rpass.draw_indexed(0..g.index_count, 0, 0..1);
                    }
                }
            }
            for (pipe, layer) in [
                (&gbuffer.opaque, &mesh.opaque),
                (&gbuffer.cutout, &mesh.cutout),
                (&gbuffer.translucent, &mesh.translucent),
            ] {
                if layer.indices.is_empty() {
                    continue;
                }
                let vbuf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&layer.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                let ibuf = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&layer.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                rpass.set_pipeline(pipe);
                rpass.set_bind_group(1, &gbuffer.atlas_bind, &[]);
                rpass.set_vertex_buffer(0, vbuf.slice(..));
                rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..layer.indices.len() as u32, 0, 0..1);
            }
        }

        // --- Deferred lighting pass -----------------------------------------
        self.ensure_lighting_bind();
        let (bind, pipe) = {
            let s = self.staged.as_ref().unwrap();
            (
                s.lighting_bind.clone().expect("lighting bind"),
                s.lighting_pipeline.clone(),
            )
        };
        let lit_view = self.scene.as_ref().unwrap().lit_view.clone();
        let color_atts = [Some(Self::color_attachment_clear(
            &lit_view,
            wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        ))];
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("deferred-lighting"),
            multiview_mask: None,
            color_attachments: &color_atts,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&pipe);
        rpass.set_bind_group(0, &bind, &[]);
        rpass.draw(0..3, 0..1);
    }

    /// Record the three-layer scene draw into `rpass` (legacy path).
    fn record_scene(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        meshes: &MeshedChunk,
        pipes: &LayerPipelines,
    ) {
        rpass.set_bind_group(0, &self.globals_bind, &[]);

        // Streaming chunks first (persistent buffers; zero per-frame alloc).
        self.record_chunks(rpass, pipes);

        // The legacy single mesh (Phase-1 validation scene API).
        for (pipe, layer) in [
            (&pipes.opaque, &meshes.opaque),
            (&pipes.cutout, &meshes.cutout),
            (&pipes.translucent, &meshes.translucent),
        ] {
            if layer.indices.is_empty() {
                continue;
            }
            let vbuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&layer.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ibuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&layer.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            rpass.set_pipeline(pipe);
            rpass.set_bind_group(1, &pipes.atlas_bind, &[]);
            rpass.set_vertex_buffer(0, vbuf.slice(..));
            rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..layer.indices.len() as u32, 0, 0..1);
        }
    }

    /// Draw every streamed chunk through one pipeline set (opaque, cutout,
    /// translucent). Buffers are persistent — no per-frame allocation.
    fn record_chunks(&self, rpass: &mut wgpu::RenderPass<'_>, pipes: &LayerPipelines) {
        for gpu in self.chunks.values() {
            for (pipe, layer) in [
                (&pipes.opaque, &gpu[0]),
                (&pipes.cutout, &gpu[1]),
                (&pipes.translucent, &gpu[2]),
            ] {
                let Some(g) = layer else { continue };
                rpass.set_pipeline(pipe);
                rpass.set_bind_group(1, &pipes.atlas_bind, &[]);
                rpass.set_vertex_buffer(0, g.vertex_buf.slice(..));
                rpass.set_index_buffer(g.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..g.index_count, 0, 0..1);
            }
        }
    }

    const SKY: wgpu::Color = wgpu::Color {
        r: 0.62,
        g: 0.8,
        b: 1.0,
        a: 1.0,
    };

    /// Draw the pending overlay: world-space outlines first (depth-tested
    /// against the scene depth) inside their own pass, then the screen-space
    /// HUD over the finished frame. Both load the target (no clear).
    fn record_overlays(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        to_window: bool,
    ) {
        let Some(pending) = self.pending_overlay.take() else {
            return;
        };
        let (fw, fh) = self.frame_dims(to_window);
        let Some(ov) = &mut self.overlay else { return };
        // Screen-size uniform (UI px → NDC); zw slots unused.
        self.queue.write_buffer(
            &ov.screen_params_buf,
            0,
            bytemuck::bytes_of(&[fw as f32, fh as f32, 0.0, 0.0]),
        );
        OverlayState::upload_frame(
            ov,
            &self.device,
            &self.queue,
            &pending.screen,
            &pending.world,
        );

        // World-space outlines + particles (depth-tested, no depth writes).
        if ov.outline_count > 0 {
            let pipe = if to_window {
                &ov.outline_pipeline
            } else {
                &ov.outline_pipeline_capture
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay-outline"),
                multiview_mask: None,
                color_attachments: &[Some(Self::color_attachment_load(target))],
                depth_stencil_attachment: depth_view.map(|v| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: v,
                        // Load the scene depth (test against the world); no writes.
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(pipe);
            rpass.set_bind_group(0, &self.globals_bind, &[]);
            rpass.set_bind_group(1, &ov.logo_bind, &[]);
            rpass.set_vertex_buffer(0, ov.outline_vbuf.slice(..));
            rpass.set_index_buffer(ov.outline_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..ov.outline_count, 0, 0..1);
        }

        // Screen-space HUD last (over everything; no depth attachment).
        if ov.screen_count > 0 {
            let pipe = if to_window {
                &ov.screen_pipeline
            } else {
                &ov.screen_pipeline_capture
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay-screen"),
                multiview_mask: None,
                color_attachments: &[Some(Self::color_attachment_load(target))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(pipe);
            rpass.set_bind_group(0, &ov.screen_bind, &[]);
            rpass.set_bind_group(1, &ov.logo_bind, &[]);
            rpass.set_vertex_buffer(0, ov.screen_vbuf.slice(..));
            rpass.set_index_buffer(ov.screen_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..ov.screen_count, 0, 0..1);
        }
    }

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

    /// Color attachment that LOADS the previous content (overlay passes).
    fn color_attachment_load<'a>(
        view: &'a wgpu::TextureView,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        }
    }

    /// Color attachment with an explicit clear color.
    fn color_attachment_clear<'a>(
        view: &'a wgpu::TextureView,
        color: wgpu::Color,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        }
    }

    /// Shared depth attachment.
    fn depth_attachment<'a>(
        view: &'a wgpu::TextureView,
    ) -> wgpu::RenderPassDepthStencilAttachment<'a> {
        wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }
    }

    fn ensure_scene(&mut self, w: u32, h: u32) {
        if let Some(s) = &self.scene {
            if s.w == w && s.h == h {
                return;
            }
        }
        let mk = |label: &'static str, fmt: wgpu::TextureFormat| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let albedo = mk("gbuffer-albedo", wgpu::TextureFormat::Rgba8UnormSrgb);
        let lightmap = mk("gbuffer-lightmap", wgpu::TextureFormat::Rgba8Unorm);
        let lit = mk("lit-color", wgpu::TextureFormat::Rgba8UnormSrgb);
        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());
        let lightmap_view = lightmap.create_view(&wgpu::TextureViewDescriptor::default());
        let lit_view = lit.create_view(&wgpu::TextureViewDescriptor::default());
        // Depth32Float + TEXTURE_BINDING: the lighting stage reads scene
        // depth back as float (world-position reconstruction). Depth24Plus
        // cannot be texture-sampled as float on all backends.
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        self.scene = Some(SceneTargets {
            albedo,
            albedo_view,
            lightmap,
            lightmap_view,
            lit,
            lit_view,
            depth_view,
            w,
            h,
        });
        self.post_bind_group = None;
        if let Some(staged) = &mut self.staged {
            staged.lighting_bind = None;
        }
    }

    /// Non-filtering sampler for the lighting stage's depth/albedo loads
    /// (textureLoad has no filtering, but wgpu still type-checks the
    /// sampler's filtering flag against the layout).
    fn lighting_sampler(&self) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lighting-load"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        })
    }

    /// (Re)build the lighting-stage bind group against the current targets.
    fn ensure_lighting_bind(&mut self) {
        let Some(staged) = &mut self.staged else {
            return;
        };
        if staged.lighting_bind.is_some() {
            return;
        }
        let Some(scene) = &self.scene else { return };
        let layout = staged.lighting_bind_layout.clone();
        let params = staged.lighting_params_buf.clone();
        let shadow_params = staged.shadow_params_buf.clone();
        let shadow_view = staged.shadow.view.clone();
        let shadow_sampler = staged.shadow_sampler.clone();
        let albedo_view = scene.albedo_view.clone();
        let lightmap_view = scene.lightmap_view.clone();
        let depth_view = scene.depth_view.clone();
        let post_sampler = self.lighting_sampler();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&post_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&lightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shadow_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });
        if let Some(staged) = &mut self.staged {
            staged.lighting_bind = Some(bind);
        }
    }

    fn rebind_post_to_lit(&mut self) {
        let Some(scene) = &self.scene else { return };
        let lit_view = scene.lit_view.clone();
        let layout = self.post_bind_layout.clone();
        let params = self.post_params_buf.clone();
        let sampler = self.post_sampler.clone();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post-bg-lit"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lit_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        self.post_bind_group = Some(bg);
    }

    fn ensure_post_bind(&mut self) {
        if self.post_bind_group.is_none() {
            let Some(scene) = &self.scene else { return };
            let view = scene.albedo_view.clone();
            let layout = self.post_bind_layout.clone();
            let params = self.post_params_buf.clone();
            let sampler = self.post_sampler.clone();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post-bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            self.post_bind_group = Some(bg);
        }
    }

    /// Record the post pass from the bound post texture into `target`.
    fn record_post_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
    ) {
        let Some(bind) = &self.post_bind_group else {
            return;
        };
        let color_atts = [Some(Self::color_attachment(target))];
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post"),
            multiview_mask: None,
            color_attachments: &color_atts,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind, &[]);
        rpass.draw(0..3, 0..1);
    }

    /// Effect configuration for the post pass.
    fn post_params(&self) -> PostParams {
        if let Some(staged) = &self.staged {
            let cfg = &staged.config;
            let post = cfg.post;
            return PostParams {
                flags: [
                    (if post.map(|p| p.bloom).unwrap_or(false) {
                        2
                    } else {
                        0
                    }) | (if post.is_some() { 4 } else { 0 }),
                    0,
                    0,
                    0,
                ],
                bloom_threshold: post.map(|p| p.bloom_threshold).unwrap_or(0.8),
                bloom_intensity: post.map(|p| p.bloom_intensity).unwrap_or(0.0),
                vignette: post.map(|p| p.vignette).unwrap_or(0.0),
                _pad: 0.0,
            };
        }
        match self.settings.quality {
            // Plain upscale: all effects off.
            RenderQuality::Low | RenderQuality::Medium => PostParams {
                flags: [0; 4],
                bloom_threshold: 0.0,
                bloom_intensity: 0.0,
                vignette: 0.0,
                _pad: 0.0,
            },
            RenderQuality::High => PostParams {
                flags: [1 | 2 | 4, 0, 0, 0],
                bloom_threshold: 1.0,
                bloom_intensity: 0.6,
                vignette: 0.25,
                _pad: 0.0,
            },
            RenderQuality::Ultra => PostParams {
                flags: [1 | 2 | 4, 0, 0, 0],
                bloom_threshold: 0.85,
                bloom_intensity: 0.9,
                vignette: 0.3,
                _pad: 0.0,
            },
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    /// The renderer's active presentation settings.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Supply the world-light grid and a mesh variant whose `light` attribute
    /// carries propagated voxel light (the deferred stage's lightmap input).
    pub fn set_world_lighting(&mut self, light: LightGrid, mesh: MeshedChunk) {
        self.world_light = Some(light);
        self.mesh_lighting = Some(mesh);
    }

    /// Insert (or replace) one streaming chunk's GPU buffers.
    pub fn set_chunk(&mut self, pos: ChunkPos, mesh: MeshedChunk) {
        let gpu = upload_meshed(&self.device, &mesh);
        self.chunks.insert(pos, gpu);
    }

    /// Drop a streamed chunk (unload / out of view distance).
    pub fn drop_chunk(&mut self, pos: ChunkPos) {
        self.chunks.remove(&pos);
    }

    /// Drop every streamed chunk's GPU buffers (world transitions: the
    /// new session's streamer re-uploads as it loads — stale meshes must
    /// never survive a menu → world switch).
    pub fn clear_chunks(&mut self) {
        self.chunks.clear();
    }

    /// Enable the day/night cycle at `fraction` (0..1: 0 sunrise, 0.25 noon,
    /// 0.5 sunset, 0.75 midnight). Pass `None` to pin the configured sun.
    pub fn set_day_fraction(&mut self, fraction: Option<f32>) {
        self.day_fraction = fraction.map(|f| f.rem_euclid(1.0));
    }

    /// Pin the sun to an absolute phase (0..1 through the day), overriding
    /// both the client's day cycle and the shader config's sun angle —
    /// the title screen's golden-hour backdrop. `None` restores normal
    /// behavior (day cycle / configured angle).
    pub fn set_sun_phase_override(&mut self, phase: Option<f32>) {
        self.sun_phase_override = phase.map(|p| p.rem_euclid(1.0));
    }

    /// Number of streamed chunks currently uploaded.
    pub fn streamed_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Is the overlay GPU state initialized (set_overlay called at least
    /// once)? The client uses this to time one-shot uploads (logo art).
    pub fn overlay_ready(&self) -> bool {
        self.overlay.is_some()
    }

    /// Queue the overlay geometry for the next frame (HUD + world-space
    /// outlines/particles). Empty lists simply draw nothing — passing an
    /// all-empty `HudDraw` every frame is valid (and hides the overlay).
    /// Lazily initializes the overlay GPU state on first call.
    pub fn set_overlay(&mut self, draw: HudDraw) {
        if self.overlay.is_none() {
            self.overlay = Some(OverlayState::new(
                &self.device,
                self.surface_config.format,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                &self.globals_bgl,
                &self.globals_bind,
            ));
        }
        // Flush art that arrived before overlay init.
        if let Some((w, h, rgba)) = self.pending_logo.take() {
            Self::upload_logo(
                self.overlay.as_mut().unwrap(),
                &self.device,
                &self.queue,
                w,
                h,
                rgba,
            );
        }
        if let Some((w, h, rgba)) = self.pending_bg.take() {
            Self::upload_bg(
                self.overlay.as_mut().unwrap(),
                &self.device,
                &self.queue,
                w,
                h,
                rgba,
            );
        }
        self.pending_overlay = Some(draw);
    }

    /// Set the overlay art texture (menu logo). RGBA8, straight alpha;
    /// sized quads sample it in UI pixels. Replaces any previous texture.
    pub fn set_logo_texture(&mut self, width: u32, height: u32, rgba: Vec<u8>) {
        let Some(ov) = &mut self.overlay else {
            // Overlay not initialized yet (no set_overlay this session):
            // stash and flush when it comes up (one-frame delay at worst).
            self.pending_logo = Some((width, height, rgba));
            return;
        };
        Self::upload_logo(ov, &self.device, &self.queue, width, height, rgba);
    }

    fn upload_logo(
        ov: &mut OverlayState,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) {
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("overlay-logo"),
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
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &rgba,
        );
        let view = tex.create_view(&Default::default());
        ov.logo_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay-tex-bg"),
            layout: &ov.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&ov.logo_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&ov.bg_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&ov.bg_sampler),
                },
            ],
        });
        ov.logo_view = view;
        ov.logo_tex = Some(tex);
    }

    /// Set the full-screen menu background art (menu_background.png).
    /// RGBA8, straight alpha; UV-mapped 0..1 across the quad.
    pub fn set_menu_background(&mut self, width: u32, height: u32, rgba: Vec<u8>) {
        let Some(ov) = &mut self.overlay else {
            self.pending_bg = Some((width, height, rgba));
            return;
        };
        Self::upload_bg(ov, &self.device, &self.queue, width, height, rgba);
    }

    fn upload_bg(
        ov: &mut OverlayState,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) {
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("overlay-menu-bg"),
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
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &rgba,
        );
        let view = tex.create_view(&Default::default());
        ov.bg_bind_rebuild(device, &view);
        ov.bg_view = view;
        ov.bg_tex = Some(tex);
    }

    /// Change quality at runtime (also used by the F4 hotkey). Clears any
    /// shader-pack configuration override; rebuilding the scene target
    /// happens lazily on the next frame.
    pub fn set_quality(&mut self, quality: RenderQuality) {
        self.settings.quality = quality;
        self.rebuild_staged();
        self.scene = None;
        self.post_bind_group = None;
    }

    /// Apply the settings of an externally-validated shader configuration
    /// (from feathered-packs). Replaces the preset until `set_quality` is
    /// called. The renderer stays pack-agnostic: this takes the *generic*
    /// effect configuration, never pack-specific data.
    pub fn apply_shader_config(&mut self, config: ShaderEffectConfig) {
        self.rebuild_staged_with(config);
        self.scene = None;
        self.post_bind_group = None;
    }

    fn rebuild_staged(&mut self) {
        let config = self.effective_config_for(self.settings.quality);
        self.rebuild_staged_with(config);
    }

    fn effective_config_for(&self, q: RenderQuality) -> ShaderEffectConfig {
        ShaderEffectConfig::for_quality(q)
    }

    /// (Re)build the staged pipeline set for `config`. A configuration with
    /// no staged effects tears the staged state down (Low preset / packs
    /// that disable everything).
    fn rebuild_staged_with(&mut self, config: ShaderEffectConfig) {
        if !config.uses_staged_pipeline() {
            self.staged = None;
            return;
        }
        if let Some(s) = &self.staged {
            if s.config == config {
                return;
            }
        }
        let device = &self.device;
        let shadow_res = config
            .shadows
            .as_ref()
            .map(|s| s.resolution)
            .unwrap_or(1024);
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d {
                width: shadow_res,
                height: shadow_res,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-cmp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::Less),
            ..Default::default()
        });

        let lighting_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lighting-params"),
            contents: &[0u8; std::mem::size_of::<LightingParams>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shadow_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow-params"),
            contents: &[0u8; std::mem::size_of::<ShadowParams>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sun_globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sun-globals"),
            contents: bytemuck::bytes_of(&Globals {
                view_proj: [[0.0; 4]; 4],
                anim_offsets: [0.0; 64],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lighting_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // Scene depth is loaded (not compared): float texture.
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        // --- Pipelines -----------------------------------------------------
        let terrain = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("terrain.wgsl").into()),
        });
        let lighting = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lighting.wgsl").into()),
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
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pl"),
            bind_group_layouts: &[Some(&globals_bgl), Some(&atlas_bgl)],
            immediate_size: 0,
        });

        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm16x2,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint8x2,
                    offset: 20,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x2,
                    offset: 22,
                    shader_location: 4,
                },
            ],
        })];

        let make_gbuffer = |blend: BlendMode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gbuffer"),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &terrain,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &terrain,
                    entry_point: Some(match blend {
                        BlendMode::Cutout => "fs_gbuffer",
                        BlendMode::Alpha => "fs_gbuffer_blend",
                    }),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            blend: Some(match blend {
                                BlendMode::Cutout => wgpu::BlendState::REPLACE,
                                BlendMode::Alpha => wgpu::BlendState::ALPHA_BLENDING,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(match blend {
                                BlendMode::Cutout => wgpu::BlendState::REPLACE,
                                BlendMode::Alpha => wgpu::BlendState::ALPHA_BLENDING,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    // Scene depth is Depth32Float (bindable for the lighting
                    // stage's depth readback).
                    format: wgpu::TextureFormat::Depth32Float,
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

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &terrain,
                entry_point: Some("vs_shadow"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &terrain,
                entry_point: Some("fs_shadow"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let lighting_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lighting-pl"),
            bind_group_layouts: &[Some(&lighting_bind_layout)],
            immediate_size: 0,
        });
        let make_lighting = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("lighting"),
                layout: Some(&lighting_pl),
                vertex: wgpu::VertexState {
                    module: &lighting,
                    entry_point: Some("vs_post"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &lighting,
                    entry_point: Some("fs_lighting"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let sun_globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sun-globals-bg"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sun_globals_buf.as_entire_binding(),
            }],
        });

        self.staged = Some(StagedState {
            gbuffer: GbufferPipelines {
                opaque: make_gbuffer(BlendMode::Cutout),
                cutout: make_gbuffer(BlendMode::Cutout),
                translucent: make_gbuffer(BlendMode::Alpha),
                atlas_bind: self.pipelines.atlas_bind.clone(),
            },
            shadow_pipeline,
            // The lighting stage always renders into the Rgba8UnormSrgb `lit`
            // target; the post pass composites to the window format.
            lighting_pipeline: make_lighting(wgpu::TextureFormat::Rgba8UnormSrgb),
            lighting_params_buf,
            shadow_params_buf,
            lighting_bind_layout,
            lighting_bind: None,
            sun_globals_buf,
            sun_globals_bind,
            shadow: ShadowMap {
                tex: shadow_tex,
                view: shadow_view,
                size: shadow_res,
            },
            shadow_sampler,
            config,
        });
    }
}

/// Normalized world-space direction of the view ray through the frustum
/// corner at NDC (x, y). Built from the camera basis (no matrix inverse:
/// the far-plane unprojection degenerates at w→0 and float precision on
/// large matrices made the GPU-side reconstruction unreliable).
fn corner_ray(camera: &Camera, ndc_x: f32, ndc_y: f32) -> [f32; 4] {
    let (sin_y, cos_y) = camera.yaw.sin_cos();
    let (sin_p, cos_p) = camera.pitch.sin_cos();
    let forward = [sin_y * cos_p, sin_p, -cos_y * cos_p];
    let right = [cos_y, 0.0, sin_y];
    let up = [-sin_y * sin_p, cos_p, cos_y * sin_p];
    let t = (camera.fov_y / 2.0).tan();
    let dir = [
        forward[0] + right[0] * (t * camera.aspect * ndc_x) + up[0] * (t * ndc_y),
        forward[1] + right[1] * (t * camera.aspect * ndc_x) + up[1] * (t * ndc_y),
        forward[2] + right[2] * (t * camera.aspect * ndc_x) + up[2] * (t * ndc_y),
    ];
    let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    [dir[0] / l, dir[1] / l, dir[2] / l, 0.0]
}

/// Inverse of `Camera::view_proj` (world ← clip): world = V⁻¹·P⁻¹·clip.
/// The shader's `world_pos()` divides by w after multiplying, so the
/// perspective "undo" is encoded projectively (see the P⁻¹ rows below).
fn inverse_view_proj(camera: &Camera) -> [[f32; 4]; 4] {
    let (sin_y, cos_y) = camera.yaw.sin_cos();
    let (sin_p, cos_p) = camera.pitch.sin_cos();
    let forward = [sin_y * cos_p, sin_p, -cos_y * cos_p];
    let right = [cos_y, 0.0, sin_y];
    let up = [-sin_y * sin_p, cos_p, cos_y * sin_p];
    let eye = camera.pos;

    // P⁻¹ acting on (ndc.x, ndc.y, z01, 1):
    //   x_view = m23·x/(a·(z01+m22)), y_view = m23·y/(f·(z01+m22)),
    //   z_view = −m23/(z01+m22), w = z01 + m22
    // with m22 = far/(near−far), m23 = far·near/(near−far), a = f/aspect.
    let f = 1.0 / (camera.fov_y / 2.0).tan();
    let a = f / camera.aspect;
    let m22 = camera.far / (camera.near - camera.far);
    let m23 = camera.far * camera.near / (camera.near - camera.far);
    let inv_p = [
        [m23 / a, 0.0, 0.0, 0.0],
        [0.0, m23 / f, 0.0, 0.0],
        [0.0, 0.0, 0.0, -m23],
        [0.0, 0.0, 1.0, m22],
    ];

    // V⁻¹: rows are the camera basis columns; the translation column solves
    // world = Rᵀ·(view − t) (verified against eye → view origin → eye).
    let d_r = dot3(right, eye);
    let d_u = dot3(up, eye);
    let d_f = dot3(forward, eye);
    let inv_view = [
        [
            right[0],
            up[0],
            -forward[0],
            right[0] * d_r + up[0] * d_u + forward[0] * d_f,
        ],
        [
            right[1],
            up[1],
            -forward[1],
            right[1] * d_r + up[1] * d_u + forward[1] * d_f,
        ],
        [
            right[2],
            up[2],
            -forward[2],
            right[2] * d_r + up[2] * d_u + forward[2] * d_f,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ];

    compose(inv_view, inv_p)
}

/// Sun ortho view-projection centered on the camera (the map follows the
/// player, as Iris/OptiFine shadow maps do). Returns (matrix, extent).
fn sun_view_proj(cam_pos: [f32; 3], sun: &[f32; 3], distance: f32) -> ([[f32; 4]; 4], f32) {
    let extent = distance.max(16.0);
    let eye = [
        cam_pos[0] + sun[0] * 100.0,
        cam_pos[1] + sun[1] * 100.0,
        cam_pos[2] + sun[2] * 100.0,
    ];
    let view = look_at(eye, cam_pos, [0.0, 1.0, 0.0]);
    // Ortho over ±extent laterally and [1..200] along the sun direction,
    // [0,1] depth (Vulkan convention) so the shader compares raw z01.
    // The look_at above follows Camera::view_proj's convention: points in
    // FRONT of the eye have NEGATIVE view z (the perspective matrix pairs
    // that with w = −z). The z row must negate: z01 = (−z_view − near)/range.
    // A positive scale here maps every in-front point to z_clip < 0, which
    // clips the entire shadow cast to an empty map (everything lit).
    let (z_near, z_far) = (1.0f32, 200.0f32);
    let proj = [
        [1.0 / extent, 0.0, 0.0, 0.0],
        [0.0, 1.0 / extent, 0.0, 0.0],
        [
            0.0,
            0.0,
            -1.0 / (z_far - z_near),
            -z_near / (z_far - z_near),
        ],
        [0.0, 0.0, 0.0, 1.0],
    ];
    (compose(proj, view), extent)
}

/// Ambient strength from sun elevation (day/night dimming). 1.0 in full
/// daylight, easing to a small moonlit floor after sunset. Smooth over the
/// dusk band so the transition never pops.
fn ambient_for_sun(elev_sin: f32) -> f32 {
    const DAY: f32 = 0.18; // elevation sin above which it is fully day
    const NIGHT: f32 = -0.12; // below which it is fully night
    let t = ((elev_sin - NIGHT) / (DAY - NIGHT)).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    0.12 + 0.88 * smooth
}

fn mip_concat(atlas: &Atlas) -> Vec<u8> {
    let mut out = atlas.pixels.clone();
    for m in &atlas.mips {
        out.extend_from_slice(m);
    }
    out
}

/// Atlas lookup table for streaming: SpriteId → (x, y, frame_w, frame_h,
/// frames, stride, fully-opaque-sprite). `'static`: owns a copy of the
/// sprite-name list and the per-entry opacity data, so the client's streamer
/// needs no borrow of the registry or atlas. Mirrors `build_meshes` exactly
/// (same opacity rule: a sprite is opaque only if every texel is opaque).
pub fn sprite_uv_table(
    registry: &Registry,
    atlas: &Atlas,
) -> std::sync::Arc<dyn Fn(u32) -> Option<(u32, u32, u32, u32, u32, u32, bool)> + Send + Sync> {
    use std::sync::Arc;
    let names: Vec<(String, String)> = registry.sprite_names().to_vec();
    // Per-entry opacity, computed once (avoids re-scanning atlas pixels per
    // lookup during streaming).
    let opaque: Vec<bool> = names
        .iter()
        .map(|name| {
            atlas
                .get(&name.0, &name.1)
                .map(|e| atlas_opaque(atlas, e))
                .unwrap_or(false)
        })
        .collect();
    let entries: Vec<Option<(u32, u32, u32, u32, u32, u32, bool)>> = names
        .iter()
        .zip(opaque)
        .map(|(name, opaque)| {
            atlas.get(&name.0, &name.1).map(|e| {
                (
                    e.x,
                    e.y,
                    e.frame_w,
                    e.frame_h,
                    e.frames,
                    e.frame_stride,
                    opaque,
                )
            })
        })
        .collect();
    Arc::new(move |sprite_id: u32| entries.get(sprite_id as usize).copied().flatten())
}

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
        Some((
            e.x,
            e.y,
            e.frame_w,
            e.frame_h,
            e.frames,
            e.frame_stride,
            opaque,
        ))
    };
    mesh_world(
        world,
        registry,
        &name_lookup,
        (atlas.width, atlas.height),
        &tint,
    )
}

/// Build a second mesh where the per-vertex `light` attribute carries
/// propagated voxel light (sky, block — levels 0..15) from `grid`, instead of
/// the Phase-1 fullbright default.
pub fn build_lighting_meshes(
    world: &World,
    registry: &Registry,
    atlas: &Atlas,
    grid: &LightGrid,
) -> MeshedChunk {
    let mut mesh = build_meshes(world, registry, atlas);
    let [sx, sy, sz] = world.size.map(|s| s as i64);
    for layer in [&mut mesh.opaque, &mut mesh.cutout, &mut mesh.translucent] {
        for v in &mut layer.vertices {
            let x = (v.pos[0].floor() as i64).clamp(0, sx - 1);
            let y = (v.pos[1].floor() as i64).clamp(0, sy - 1);
            let z = (v.pos[2].floor() as i64).clamp(0, sz - 1);
            v.light = grid.get(x, y, z).unwrap_or([15, 0]);
        }
    }
    mesh
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
