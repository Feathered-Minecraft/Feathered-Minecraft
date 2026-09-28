# Security Policy

Feathered Minecraft is a local-first client and engine: it talks to no
game server, sends no telemetry, and never downloads assets silently. Most
attack surface is therefore local — resource-pack and shader-pack import,
world save loading, and the asset compiler — and that is where security
reports are most valuable.

## Supported versions

The project is in early development and ships no tagged releases yet. Only
the current `main` branch receives security fixes; please verify a reported
issue against a fresh checkout of `main` before reporting.

## Reporting a vulnerability

**Do not open a public GitHub issue, pull request, or discussion for
security problems.**

Report privately by e-mail to:

**amandeep.intl+featheredminecraft@gmail.com**
(subject line: `Feathered security report`)

If private vulnerability reporting is enabled for this repository, you may
instead use GitHub's *Security → Advisories → Report a vulnerability* flow,
which keeps the report confidential.

You will receive an acknowledgment as soon as practical and progress updates
while the issue is investigated. Please give a reasonable window for a fix
and coordinated disclosure before making any details public — we ask that
you do not publicly disclose the vulnerability or exploitation details until
a fix is available and disclosure is agreed.

## What to include

* The exact commit, branch, or build you tested (and how you built it).
* Your environment: OS version, GPU/driver where relevant, Rust toolchain
  version.
* Precise steps to reproduce, including the exact command and arguments.
* Observed behavior versus expected behavior (panic backtraces, logs, crash
  output are all helpful).
* Your assessment of impact and, if known, the affected subsystem (pack
  import, shader-pack parsing, save loading, asset compiler, renderer, …).
* Whether you would like credit, and the name or handle to use.

## What not to do

* Do not publicly post the issue or any exploit details before coordinated
  disclosure is agreed.
* Do not test against accounts, worlds, or data you do not own.
* Do not go beyond what is needed to demonstrate the issue — a minimal,
  safe proof of concept is ideal.
* Do not attach Minecraft assets to your report. Describe the pack or save
  instead; Mojang/Microsoft content must never enter this repository,
  including via security reports.
* Do not submit resource packs or shader packs whose licenses forbid
  redistribution; describe their structure or attach a minimal,
  license-clean reproduction.

## Scope

**In scope:** memory safety, panics or hangs triggered by crafted
resource packs, shader packs, or world saves; path traversal or file
overwrite during pack import/extraction (including zip-slip); cache
poisoning or digest confusion in the asset compiler; unsafe code defects;
anything that lets untrusted input escape its sandbox.

**Out of scope:** bugs in imported third-party packs themselves, issues
that require physical access or a trusted local attacker with arbitrary
code execution already, and Minecraft server/protocol exploitation (the
client does not implement one). Panics caused by clearly malformed input
are still worth reporting — they are robustness bugs even when they are
not exploitable.

## Licensing note

Security fixes follow the same rules as all other contributions:
GPL-3.0, no Mojang assets, no license-incompatible code.
