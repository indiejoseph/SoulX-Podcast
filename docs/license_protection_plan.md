# License protection & encrypted on-prem deployment — plan

Goal: deploy the SoulX-Podcast API (`docker-compose.deploy.yml` stack) to
customers' on-prem servers with (a) the model weights protected against
copying, and (b) license enforcement bound to specific machines with expiry.

Two protection tiers sharing one license server. The tier is a per-license
policy decision, not a separate codebase.

| | Tier 1 — encrypted weights + fingerprint | Tier 2 — confidential computing |
|---|---|---|
| Protects against | image/volume copying, casual insiders, unlicensed redeploy | everything in Tier 1 **plus** root-level RAM/VRAM dump and license-check patching |
| Customer hardware | any NVIDIA GPU (3090/4090/A-series OK) | **H100 / H200 / Blackwell** + SEV-SNP (EPYC Milan+) or TDX (Xeon SPR+) CPU |
| Machine binding | self-reported fingerprint (GPU UUID, MAC, machine-id) — spoofable by root | hardware-signed remote attestation — unforgeable, also measures our code |
| Residual risk | root memory dump during runtime → contract/legal + watermark covers | key compromise on our license server; attestation infra bugs |

---

## Architecture (shared across tiers)

```
┌────────────────────────── customer on-prem ──────────────────────────┐
│  Docker container (Tier 1) / Confidential VM (Tier 2)                │
│                                                                      │
│  entrypoint                                                          │
│    1. collect evidence:                                              │
│         T1: fingerprint {gpu_uuid, mac, machine_id}                  │
│         T2: CPU TEE report + GPU CC attestation report               │
│    2. POST evidence + license_id  ──────────────┐                    │
│    5. decrypt weights IN MEMORY, start API      │                    │
│    6. heartbeat re-validation every N hours     │                    │
└─────────────────────────────────────────────────┼────────────────────┘
                                                  │ TLS
┌───────────────────────── our license server ────▼────────────────────┐
│    3. verify evidence:                                               │
│         T1: fingerprint matches licensed machine record              │
│         T2: verify cert chains (AMD/Intel + NVIDIA NRAS),            │
│             measurement ∈ released-image hashes, GPU UUID licensed   │
│    4. policy OK + unexpired → release per-customer AES key           │
│       (T2: key wrapped to the enclave's public key)                  │
└───────────────────────────────────────────────────────────────────────┘
```

Weights (`model.safetensors`, `flow.pt`, `hift.pt`, `composer.pt`,
`campplus.onnx`) ship AES-256-GCM-encrypted inside the image or the HF
download. Plaintext never touches disk — decrypt straight into
`safetensors.torch.load(bytes)` / `torch.load(io.BytesIO(...))`.

---

## Phase A — encryption tooling + in-memory loader (Tier-1 core, ~2-3 days)

Everything here is also required by Tier 2, so build it first.

1. **`scripts/licensing/encrypt_model.py`**
   - Input: a model dir (e.g. `tmp/meanflow_awq_model` + `composer.pt`).
   - Per-customer AES-256-GCM key (random, stored in the license server DB).
   - Output: `<name>.enc` per weight file + `manifest.json`
     (file list, nonces, SHA-256 of plaintexts, key_id, customer_id).
   - Tokenizer/config JSONs stay plaintext (no IP value, needed pre-decrypt).

2. **Weight watermarking (in the same script)**
   - Before encryption, flip a per-customer pattern in the low-order
     mantissa bits of a fixed, documented subset of `flow.pt` tensors
     (~few thousand values, ε ≤ 1e-6 — inaudible).
   - Record the pattern in the license DB. If weights ever leak, the
     pattern identifies the customer.

3. **`soulxpodcast/licensing/loader.py` — in-memory decrypting loader**
   - `load_encrypted_state_dict(path_enc, key) -> dict[str, Tensor]`
   - Hook points (keep them minimal and central):
     - LLM: vLLM loads from a directory path, so the LLM safetensors is
       the hard case — options, decided in this phase:
       a. decrypt to a **tmpfs** mount (`/dev/shm/model`), point vLLM at
          it, `shred` after engine init completes. RAM-backed, never on
          disk, but plaintext exists in page cache for ~30 s. Acceptable
          for Tier 1 (Tier 2 encrypts that RAM anyway). **Default.**
       b. custom vLLM model-loader plugin streaming decrypted tensors —
          cleaner but multi-day vLLM-internals work. Only if (a) proves
          unacceptable.
     - flow/hift/composer/campplus: direct in-memory decrypt (torch +
       onnxruntime both accept bytes). No disk at all.
   - `SoulXPodcastService._load_model` gains an
     `if manifest.json present → licensed path` branch; unencrypted
     checkpoints keep working unchanged (dev workflow untouched).

4. **Tests**: round-trip encrypt→load equality vs plaintext checkpoint
   (bit-exact state dict); tmpfs path leaves no residue after startup;
   tampered ciphertext → hard fail.

## Phase B — license server + client (Tier 1 complete, ~3-4 days)

1. **License server** (small FastAPI + SQLite/Postgres, runs on our infra)
   - `POST /v1/activate` — body: `{license_id, evidence}`. Verifies, returns
     `{wrapped_key, license_token(JWT, exp), heartbeat_interval}`.
   - `POST /v1/heartbeat` — extends the token; server can revoke.
   - Admin CLI: issue license (customer, machine fingerprint or "TOFU on
     first activate", expiry, tier), revoke, list activations.
   - Evidence verification is pluggable: `FingerprintVerifier` (Tier 1)
     now, `AttestationVerifier` (Tier 2) in Phase C.
   - TOFU note: first activation records the fingerprint; later
     activations must match. Avoids collecting fingerprints pre-sale.

2. **Client in the entrypoint** (`scripts/docker_entrypoint.sh` →
   thin Python `soulxpodcast/licensing/client.py`)
   - Collect fingerprint: GPU UUID (`nvidia-smi --query-gpu=uuid`),
     primary MAC, `/etc/machine-id`.
   - Activate → hold key in memory only → hand to loader.
   - Heartbeat thread: revalidate every `heartbeat_interval` (default 6 h),
     grace window 48 h for network outages, then stop accepting new
     requests (finish in-flight, exit non-zero → `restart: unless-stopped`
     retries activation).
   - Env: `LICENSE_SERVER_URL`, `LICENSE_ID`, `LICENSE_OFFLINE_GRACE_H`.

3. **Deploy packaging**
   - `docker-compose.customer.yml`: like `docker-compose.deploy.yml` but
     model volume holds `.enc` files + manifest; adds `LICENSE_*` env;
     drops `HF_TOKEN` (weights delivered encrypted, via HF private repo
     or a tarball — either is fine since they're ciphertext).
   - **Nuitka-compile** `soulxpodcast/licensing/` + `api/` into `.so` in
     `Dockerfile.serve` (new build stage, `LICENSED=true` build arg) so
     the license client isn't a trivially editable `.py`. Deterrence
     only — documented as such.

4. **Tests**: expired license → refuses to start; revoked mid-run → stops
   after grace; fingerprint mismatch → refused; clock rollback on client
   → caught by server-side timestamps.

## Phase C — remote attestation (Tier 2, ~1-2 weeks incl. hardware access)

Blocked on access to an H100/H200 + SEV-SNP/TDX host. Nothing in Phase
A/B needs rework — this swaps evidence + adds key wrapping.

1. **PoC first, standalone** (no SoulX code): run NVIDIA `nvtrust`
   end-to-end sample on the CC host — CVM boot, GPU CC mode on, CPU +
   GPU attestation verified via NRAS. Exit criterion: verified
   attestation bundle on our laptop. This de-risks everything else.

2. **`AttestationVerifier`** in the license server
   - Verify AMD-SP / Intel certificate chain on the CPU report.
   - Verify GPU report via NVIDIA NRAS API (hosted) — self-hosted
     verifier later if a customer requires no-NVIDIA-dependency.
   - Policy: `measurement ∈ allowed_measurements(release)`,
     `gpu_uuid ∈ license.machines`, CC mode == ON.
   - **Measurement registry**: CI publishes the CVM image measurement per
     release; license server admin approves it. Every image update = new
     measurement (this is the main ongoing ops cost of Tier 2).

3. **Key wrapping**: enclave generates an ephemeral keypair, public key
   goes into the attestation report's user-data field; server encrypts
   the AES key to it. Host never sees an unwrappable key even on the wire.

4. **CVM guest image**: Ubuntu 24.04 CVM kernel + NVIDIA CC driver +
   docker; our existing container runs unmodified inside it. Deliver as
   a qcow2 + install doc for the customer's hypervisor.

5. **Perf check**: re-run `bench_stream.py` / `bench_concurrent.py`
   inside the CVM — expect a few % (encrypted PCIe bounce buffers);
   verify RTF targets still hold on H100.

## Phase D — productization (ongoing)

- License admin UI (or just CLI + runbook to start).
- Structured audit log on the license server (who activated where, when).
- Contract templates referencing the watermark + audit rights (legal, not
  code — flag to counsel).
- Per-release key rotation policy; incident runbook for suspected leaks
  (watermark extraction procedure).

---

## Decisions taken (revisit if requirements change)

- **One license server, two evidence types** — tier is policy, not fork.
- **tmpfs decrypt for the vLLM safetensors** in Tier 1 rather than a
  custom vLLM loader — pragmatic; Tier 2 removes the residual exposure.
- **TOFU fingerprint enrollment** — avoids pre-sale fingerprint exchange.
- **Watermark all tiers** — the cheap universal deterrent; survives even
  a successful RAM dump.
- **Nuitka over PyArmor/etc.** — real compilation, no vendor runtime.

## Open questions (need answers before Phase B ships)

1. Offline-only customers (air-gapped)? → would need signed license
   *files* + hardware dongle or manual reactivation ritual; currently
   out of scope, phone-home assumed.
2. Who hosts the license server, and what SLA? An outage + expired grace
   window stops customer prod. Grace default 48 h — confirm with the
   first customer's ops.
3. Is the HF private repo still the delivery channel for encrypted
   weights, or tarball-on-portal? (Ciphertext either way; HF is easier
   to reuse — `docker_entrypoint.sh` already downloads from HF.)
4. First Tier-2 customer/hardware timeline — determines whether Phase C
   starts now or stays parked after the PoC.

## Effort summary

| Phase | Scope | Estimate |
|---|---|---|
| A | encrypt tool + watermark + in-memory loader + tests | 2–3 days |
| B | license server + client + compose + Nuitka + tests | 3–4 days |
| C | attestation PoC + verifier + CVM image + bench | 1–2 weeks (needs H100 CC host) |
| D | admin/ops/legal hardening | ongoing |

Tier 1 shippable end of Phase B (~1 week). Tier 2 additive on top.
