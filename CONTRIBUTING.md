# Contributing to Feathered Minecraft

Thank you for considering a contribution. Feathered is a native
Minecraft-compatible client and engine written in Rust and rendered with
wgpu. It is a from-scratch independent implementation: it ships no game
assets, collects no telemetry, and stays under **GPL-3.0**.

By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

You need a stable Rust toolchain (install via [rustup](https://rustup.rs)).
Windows and Linux are the primary targets.

```sh
git clone https://github.com/Feathered-Minecraft/Feathered-Minecraft.git
cd Feathered-Minecraft
cargo build --release
```

The binary is `target/release/feathered` (`.exe` on Windows). On first
launch run `feathered first-run` to import a resource pack — Feathered
never downloads or bundles Minecraft assets itself, and contributions must
keep it that way.

## Project layout

The repository is a Cargo workspace with a strict dependency direction
(`app → client → renderer → chunk → world → assets`, with `packs` alongside
and `engine/entity` below `client`). The
[README](README.md#project-layout) documents each crate's purpose. Put new
code in the right crate and never introduce a dependency that points
upwards. Large features usually belong in a new `engine/` crate, wired into
the client through a small surface.

## Building and testing

```sh
cargo check --workspace             # fast compile validation
cargo test --workspace              # the full test suite
cargo build --release               # what you actually ship/run
```

Please make sure `cargo check --workspace` and `cargo test --workspace`
pass before opening a PR, and add tests for anything you fix or add:

* world generation is **deterministic per seed** and must stay correct for
  negative chunk coordinates — extend the worldgen tests if you touch it;
* chunk streaming and rendering changes should not regress the golden
  mesher/asset tests;
* GPU-dependent checks live behind `#[ignore]` and are run explicitly, e.g.
  the title-screen preview:

  ```sh
  cargo test -p feathered-client --test menu_preview -- --ignored --nocapture
  ```

Keep the engine light on low-end hardware: budget background work, avoid
per-frame allocations in hot paths, and bound anything that could spike
(generation, meshing, AI, pathfinding). If your change affects
performance, include measurements from a release build, not guesses.

## Formatting

Format all Rust code with rustfmt and keep the code warning-clean:

```sh
cargo fmt --all
cargo fmt --all -- --check          # what CI-style checks look for
cargo clippy --workspace --all-targets
```

Do not introduce new compiler or clippy warnings.

## Reporting issues

Search [existing issues](https://github.com/Feathered-Minecraft/Feathered-Minecraft/issues)
first, then open one issue per report using the provided templates:

* **Bug report** — include your OS, GPU/driver, Rust version, the exact
  command you ran, and steps to reproduce. Crash output or logs are gold.
* **Feature request** — explain the problem you are solving and how it fits
  Feathered's goals: native performance, no bundled assets, no telemetry.
* **Performance report** — include the release-build numbers you measured
  and the hardware they came from.

Do **not** open a regular issue for security vulnerabilities — see
[SECURITY.md](SECURITY.md) for private reporting.

## Pull requests

1. Fork and create a focused branch (`feature/…` or `fix/…`).
2. Keep PRs small and single-purpose; a PR should do one thing well.
3. Fill in the [pull request template](.github/pull_request_template.md):
   summary, motivation, changes, testing performed, and screenshots or
   runtime validation notes for visual/behavioral changes.
4. Link any related issues (`Fixes #123`).
5. All checks green, formatting applied, no new warnings.

For visual changes (UI, rendering), a screenshot or short capture of the
before/after is strongly appreciated.

## Licensing and provenance rules

These rules are non-negotiable and protect every contributor:

* Your contributions are licensed to the project under **GPL-3.0**, matching
  [LICENSE](LICENSE). You keep copyright on what you write.
* **No Minecraft assets.** Never commit textures, sounds, models, or any
  other Mojang/Microsoft content, and never add code that downloads them
  silently. Resource packs and shader packs imported by users remain under
  their own licenses — Feathered does not relicense them.
* **Respect third-party licenses.** If you port an algorithm or idea from
  another project, check its license first. Code copied from AGPL sources
  cannot be merged into Feathered without changing Feathered's licensing,
  so it is not accepted. When in doubt, reimplement independently and note
  the inspiration in your PR.
* **Keep reference material out of the repo.** Local copies of other
  projects, extracted vanilla packs (`texture/`), test worlds (`world/`,
  `worlds/`), caches, binaries, and screenshots must stay git-ignored — the
  [.gitignore](.gitignore) already covers them; do not work around it.
* Do not remove or alter existing license headers or files.

If a contribution cannot meet these rules, it cannot be merged — but a
clean reimplementation usually can. When unsure, open an issue and ask
before writing code.
