# Feathered Minecraft

A high-performance, native Minecraft-compatible client and engine, rebuilt
from scratch for speed, efficiency, and modern hardware. Written in Rust,
rendered with wgpu.

Feathered does **not** ship with game assets and never downloads them
silently. You provide a resource pack — legally obtained — and Feathered
compiles its blockstates, models, and textures into its own binary runtime
format (no JSON is parsed at runtime). On first launch, `feathered first-run`
walks you through importing a pack.

## Project layout

The code is a seven-crate Cargo workspace with a strict dependency direction
(`app → client → renderer → chunk → world → assets`, with `packs` alongside):

| Crate | Path | Purpose |
|---|---|---|
| `feathered-app` | `app/` | `feathered` CLI: pack/shader management, first-run, compile, render |
| `feathered-client` | `client/` | Window, input, fly camera, validation scene |
| `feathered-renderer` | `engine/renderer/` | wgpu pipelines (opaque / cutout / translucent) + post-process |
| `feathered-chunk` | `engine/chunk/` | Chunk mesher: culling, variants, tinting |
| `feathered-world` | `engine/world/` | Block registry, blockstate resolution |
| `feathered-packs` | `packs/` | Resource-pack & shader-pack managers, settings, quality presets |
| `feathered-assets` | `assets/` | Pack discovery + asset compiler + atlas + cache |

The asset compiler resolves the pack's full chain — blockstate → model →
parent chain → texture → baked rotations → texture atlas — and serializes
the result into a binary `FEAT` cache with a content digest, so the client
loads pre-baked geometry definitions instead of ever touching JSON.

## Building

Requires a stable Rust toolchain.

```sh
cargo build --release
```

## Resource packs

Feathered manages packs independently of the engine. Installed packs live in
the OS data directory (`%APPDATA%\feathered\packs` on Windows,
`~/.local/share/feathered/packs` on Linux, override with
`FEATHERED_DATA_DIR`); original files are never modified.

```sh
# First launch: guided setup (import / open folder / browse online sources)
feathered first-run

feathered packs list                     # installed packs + active pack
feathered packs import <zip-or-folder>   # ZIPs are extracted; folders referenced in place
feathered packs use <id>                 # switch the active pack
feathered packs uninstall <id>           # remove (originals always untouched)
feathered packs open-folder              # open the pack data folder
feathered packs browse                   # external pack sites (CurseForge, Modrinth, ...)
feathered packs scan <dir>               # bulk-import e.g. .minecraft/resourcepacks
```

Packs are validated on import (`pack.mcmeta` + namespace folders), the
`pack_format` is detected and mapped to a Minecraft release, and the selected
pack is compiled through the asset compiler into the binary cache. Switching
packs recompiles only when needed — compiled caches are content-digested and
rebuilt automatically when stale.

## Shader packs

Shader packs are **independent** of resource packs: they change how the world
is rendered, not what is rendered.

```sh
feathered shaders list|import <zip-or-folder>|enable <id>|disable|uninstall <id>
```

Feathered detects the pack's layout (OptiFine-style `*.fsh`/`*.vsh` or
GLSL-pass style), reads its `shaders.properties` compatibility claims, and
preserves the pack's own license files verbatim. **Third-party shader packs
are never relicensed as Feathered code** — their licenses apply to them, and
Feathered's GPL-3.0 applies to Feathered. The rendering pipeline is modular:
packs configure it; none is hardcoded into the engine.

## Quality settings

Four presets scale the renderer from low-end hardware up to max quality
(`--quality low|medium|high|ultra`, or **F4** in-game to cycle). High/Ultra
run the staged deferred pipeline (per-effect toggles documented in
[docs/SHADER_COMPATIBILITY.md](docs/SHADER_COMPATIBILITY.md)):

| Preset | Scene resolution | Pipeline |
|---|---|---|
| Low | 50% (upscaled) | direct |
| Medium (default) | 100% | deferred lighting (water effects) |
| High | 100% | + shadows, AO, atmosphere, clouds, fog, exposure, post |
| Ultra | 100% | same stages, heavier per-effect configuration |

## Tests

The workspace carries golden tests that encode ground truth from the
resource pack (blockstate resolution, model parent chains, rotation baking,
atlas packing), mesher tests (culling, variant selection, layering), and
pack-manager tests (zip/folder import, version detection, invalid packs,
switching, cache generation, shader discovery, zip-slip protection):

```sh
cargo test --workspace
```

## License

Copyright © 2026 Feathered Minecraft Contributors.

Licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE).

## Note about Minecraft assets

This repository does not include, host, or download any Minecraft assets.
`texture/` (if you create it locally) holds an extracted copy of the vanilla
resource pack; its contents remain the property of Mojang/Microsoft and are
governed by the [Minecraft EULA](https://www.minecraft.net/en-us/eula). The
directory is git-ignored and must not be redistributed through this
repository.

The same policy applies to third-party content: resource packs and shader
packs you import remain under **their own licenses** — Feathered records
their source and license files for attribution but never claims, changes, or
relicenses them. Feathered itself is and remains GPL-3.0.
