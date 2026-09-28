<!--
Thank you for contributing to Feathered Minecraft!

Keep PRs focused: one logical change per pull request. Sections marked
"if applicable" can be deleted when they do not apply.
-->

## Summary

<!-- One or two sentences: what does this PR do? -->

## Motivation

<!-- Why is this change needed? Link related issues with "Fixes #123" or "Refs #45". -->

## Changes

<!-- A short list of the concrete changes. Highlight anything reviewers should look at first. -->

-

## Testing

<!-- How was this verified? Paste the commands you ran and their results. -->

- [ ] `cargo check --workspace`
- [ ] `cargo test --workspace`
- [ ] `cargo build --release`
- [ ] New or updated tests cover the changed behavior

## Screenshots / runtime validation

<!-- For visual or behavioral changes: before/after screenshots, captures, or a
     description of the runtime validation you performed (scene run, GPU preview
     test, soak test). Delete this section if not applicable. -->

## Performance notes

<!-- If this touches hot paths (rendering, meshing, worldgen, streaming, entities,
     AI), include release-build measurements and compare against the previous
     behavior. Delete if not applicable. -->

## Breaking changes

<!-- Does this change existing public APIs, save-file format, controls, CLI
     flags, or pack behavior? If yes, explain the impact and migration path.
     Write "None" if there are none. -->

## Licensing and provenance

<!-- Feathered is GPL-3.0 and never ships Mojang assets. See CONTRIBUTING.md. -->

- [ ] No Minecraft/Mojang assets are added or downloaded by this change.
- [ ] Any ported algorithm or idea is license-compatible; no source was copied
      from AGPL-licensed or otherwise incompatible projects (or it is noted and
      justified below).
- [ ] No reference trees, extracted packs (`texture/`), worlds, caches, or
      binaries are committed.
- [ ] New third-party dependencies are license-compatible and recorded in
      `Cargo.toml` only if genuinely needed.

## Checklist

- [ ] `cargo fmt --all` has been applied.
- [ ] No new compiler or clippy warnings.
- [ ] Commit messages describe the *why*, not just the *what*.
- [ ] I agree to my contribution being licensed under the project's
      [GPL-3.0 license](https://github.com/Feathered-Minecraft/Feathered-Minecraft/blob/main/LICENSE).
