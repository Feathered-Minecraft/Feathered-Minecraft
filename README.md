# Feathered Minecraft

A high-performance, native Minecraft-compatible client and engine, rebuilt
from scratch for speed, efficiency, and modern hardware. Written in Rust,
rendered with wgpu.

Feathered does **not** ship with game assets. Point it at an extracted copy of
the vanilla resource pack (or any compatible resource pack) and it compiles
the pack's blockstates, models, and textures into its own binary runtime
format — no JSON is parsed at runtime.

## Project layout

The code is a six-crate Cargo workspace with a strict dependency direction
(`app → client → renderer → chunk → world → assets`):

| Crate | Path | Purpose |
|---|---|---|
| `feathered-app` | `app/` | `feathered` CLI: pack compilation, validation, render |
| `feathered-client` | `client/` | Window, input, fly camera, validation scene |
| `feathered-renderer` | `engine/renderer/` | wgpu pipelines (opaque / cutout / translucent) |
| `feathered-chunk` | `engine/chunk/` | Chunk mesher: culling, variants, tinting |
| `feathered-world` | `engine/world/` | Block registry, blockstate resolution |
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

## Usage

By default, `feathered` looks for an extracted resource pack at
`./texture/assets` (a folder containing `pack.mcmeta`) and stores its
compiled cache in `./target/feathered-cache.bin`. Both locations can be
overridden with `--pack-dir` and `--cache`/`--out`.

```sh
# Compile the full resource pack into the binary cache (~35 MB)
cargo run --release -- compile-pack

# Compile only the ten validation blocks (fast iteration)
cargo run --release -- compile-pack --required

# Run the milestone validation checks against the compiled cache
cargo run --release -- validate

# Launch the ten-block validation scene (fly camera: WASD + Space/Shift, E to grab mouse)
cargo run --release -- render

# Render the scene headlessly to a PNG screenshot
cargo run --release -- render --screenshot shot.png
```

All commands accept `--pack-dir <dir>` to point at a different extracted
resource pack location.

## Tests

The workspace carries golden tests that encode ground truth from the
resource pack (blockstate resolution, model parent chains, rotation baking,
atlas packing) plus mesher tests (culling, variant selection, layering):

```sh
cargo test --workspace
```

## License

Copyright © 2026 Feathered Minecraft Contributors.

Licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE).

## Note about Minecraft assets

This repository does not include any Minecraft assets. `texture/` (if you
create it locally) holds an extracted copy of the vanilla resource pack, and
its contents remain the property of Mojang/Microsoft and are governed by the
[Minecraft EULA](https://www.minecraft.net/en-us/eula). The directory is
git-ignored and must not be redistributed through this repository.
