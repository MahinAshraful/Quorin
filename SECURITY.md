# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in `quorin`, please report it
**privately** rather than opening a public GitHub issue.

**Reporting channel:** open a private vulnerability report via GitHub
Security Advisories at
<https://github.com/MahinAshraful/Quorin/security/advisories/new>, or
email **mahinashraful08@gmail.com** with the subject line
`[quorin security] <summary>`.

Include:
- A description of the issue and the affected version(s).
- A minimal reproduction (input that triggers the issue, expected vs.
  observed behavior).
- Your assessment of impact (data corruption, information disclosure,
  denial-of-service, etc.) — best-effort; we'll re-triage.
- Any suggested mitigation if you have one.

We aim to acknowledge new reports within **2 business days** and to
ship a fix or coordinated disclosure plan within **30 days** for
confirmed issues.

## Trust model

`quorin` is a **single-machine** library. Its threat model assumes:

1. The process running `quorin` is trusted code; schema authors,
   operators, and producers are authenticated and authorized at the
   application layer.
2. Redis is reachable via a trusted network path. The library does
   **not** ship authentication or transport encryption — see
   [docs/operations.md](docs/operations.md) for the operator runbook
   on Redis AUTH + TLS.
3. The local `/dev/shm` filesystem and Parquet partition directories
   are protected by OS-level file permissions; cross-tenant isolation
   is the operator's responsibility.

In short: `quorin` is **not** a network-trust-boundary library. Wrap
it in your own authentication/authorization layer before exposing
producer or consumer to untrusted callers.

## What's in scope

- Memory safety of the shared-memory layer (segment header parsing,
  string-pool bounds, slot-table integrity).
- Path-traversal in Parquet partition path construction (CR.C.1
  defense-in-depth landed in v0.1.1).
- Trust-boundary validators on schema names, capacity, and
  `max_id_bytes` (CR.A.6, CR.H.4, CR.H.5).
- Supply-chain hygiene: SHA-pinned GitHub Actions, least-privilege
  workflow tokens, dependency pin discipline.

## What's out of scope

- Network-level attacks against Redis (use AUTH+TLS yourself).
- Resource exhaustion via maliciously-crafted schemas (see operator
  runbook for capacity caps).
- Side-channel attacks based on segment timing.
- Distributed-system failure modes that aren't single-machine.

## Acknowledgements

Reporters who confirm a real issue will be credited in the release
notes (with their permission) and in `CHANGELOG.md`.
