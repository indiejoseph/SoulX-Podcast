# P-EAGLE vLLM Experiment

This is experimental plumbing for trying a trained Speculators P-EAGLE
checkpoint with the SoulX vLLM inference path.

## Status

- The production image still uses the patched vLLM `0.10.1` runtime because it
  carries the SoulX RAS sampler patch.
- P-EAGLE is a newer Speculators/vLLM feature and is not expected to work on
  that default runtime.
- `VLLM_SPECULATIVE_CONFIG` is therefore opt-in. If the installed vLLM does not
  expose `EngineArgs.speculative_config`, startup fails with a clear error.
- Newer vLLM runtimes do not carry the SoulX `SamplingParams` RAS patch. For
  P-EAGLE runs, this repo installs a vLLM V1 model-owned Qwen3 sampler hook
  based on the vllm-omni CosyVoice3 pattern. Set `SOULX_VLLM_MODEL_RAS=0` only
  for an explicit no-RAS ablation.

## Runtime

Check the active environment first:

```bash
python scripts/peagle/check_vllm_speculative_support.py
```

After training a P-EAGLE checkpoint in Speculators format, write a config:

```bash
python scripts/peagle/write_speculative_config.py \
  --speculator-model /path/to/peagle/checkpoints/checkpoint_best \
  --num-speculative-tokens 4 \
  --output exports/peagle/speculative_config.json
```

Set `--num-speculative-tokens` to the P-EAGLE `num_depths` used for training.
The H100 PJM defaults use `NUM_DEPTHS=4`. The generated config defaults to
`method=eagle3` and `parallel_drafting=true`, matching the vLLM 0.22 serving
image.

Run the non-MTP vLLM path against a vLLM/speculators runtime that supports
`speculative_config`:

```bash
LLM_ENGINE=vllm \
ENABLE_MTP=false \
VLLM_USE_V1=1 \
SOULX_VLLM_MODEL_RAS=1 \
VLLM_SPECULATIVE_CONFIG=exports/peagle/speculative_config.json \
python scripts/inference/inference_test.py vllm \
  --vllm-speculative-config exports/peagle/speculative_config.json \
  --no-dialect-prompt \
  --json-output outputs/bench/inference_vllm_peagle.json
```

For API serving:

```bash
LLM_ENGINE=vllm \
ENABLE_MTP=false \
VLLM_USE_V1=1 \
SOULX_VLLM_MODEL_RAS=1 \
VLLM_SPECULATIVE_CONFIG=/app/exports/peagle/speculative_config.json \
docker compose up --build
```

The model-owned sampler keeps the public vLLM `SamplingParams` stock for the
newer runtime. Internally it applies vLLM's normal processors for the candidate
sample, checks the recent output window, and falls back to the raw/full logits
distribution when RAS fires, matching the existing SoulX HF and patched-vLLM
behavior.

## Training Shape

Clone Speculators and install the separate requirements:

```bash
scripts/peagle/setup_speculators_env.sh
```

This creates `${PROJECT_DIR}/third_party/speculators` and
`${PROJECT_DIR}/.speculators_venv` by default. Installing the `speculators`
Python package alone is not enough for this wrapper; the upstream source
checkout is needed because the wrapper calls `scripts/launch_vllm.py`,
`scripts/data_generation_offline.py`, and `scripts/train.py` from that checkout.
The setup helper installs the checkout in editable mode and the wrapper
prepends `SPECULATORS_ROOT/src` and `SPECULATORS_ROOT` to `PYTHONPATH` so those
scripts import matching source modules instead of a stale PyPI package.
It also applies an idempotent local compatibility patch for online hidden-state
training: PyTorch samplers can pass `numpy.int64` indices, while Hugging Face
`Dataset.__getitem__` requires a plain Python `int`, and the vLLM hidden-state
connector can return a temporary `.safetensors` path before it is visible to
the training worker.

The implemented offline path is:

1. Prepare SoulX `text` / `speech_tokens` / `lang` rows into Speculators Arrow format.
2. Launch the vLLM hidden-state extraction server for the verifier.
3. Generate cached hidden states with upstream `data_generation_offline.py`.
4. Train with upstream `scripts/train.py --speculator-type peagle`.
5. Write a `VLLM_SPECULATIVE_CONFIG` JSON for this repo.

On large training nodes, you can skip the separate hidden-state extraction pass
and let Speculators request teacher hidden states during training:

```bash
python scripts/peagle/train_soulx_peagle.py \
  --stage online \
  --speculators-root third_party/speculators \
  --model-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --dataset-path data/your_full_dataset \
  --work-dir outputs/peagle_soulx \
  --endpoint http://localhost:8000/v1 \
  --overwrite-preprocessed
```

`--stage online` runs prepare, train, and write-config. It sets
`--on-missing generate --on-generate delete`, so hidden states are generated
from the running vLLM endpoint and then discarded. Use `--on-generate cache`
for a hybrid first epoch that stores generated states for later reuse.

For the PJM/H100 cluster, `scripts/peagle/submit_peagle_online_h100.pjm`
prepares the dataset, starts the hidden-state server in the background, waits
for `/v1/models`, trains with online hidden-state generation, and writes the
runtime speculative config:

```bash
pjsub scripts/peagle/submit_peagle_online_h100.pjm
```

Optional `TARGET_LAYER_IDS` should list only the three EAGLE auxiliary layers,
for example `TARGET_LAYER_IDS="2 14 25"`. Do not include the verifier final
layer; the hidden-state extraction launcher appends it for target logits, while
the drafter config must keep only the three auxiliary layers.

On this PJM cluster, inline shell assignments such as
`WANDB=1 pjsub scripts/peagle/submit_peagle_online_h100.pjm` are not propagated
into the batch job. Treat the PJM file itself as the source of truth for job
settings. Edit the defaults near the top of
`scripts/peagle/submit_peagle_online_h100.pjm` before submitting.

The checked-in H100 defaults are:

```bash
DRAFT_VOCAB_SIZE=6562
AUTO_RESET_INCOMPATIBLE_CHECKPOINTS=1
WANDB=1
LOGGER=wandb
WANDB_PROJECT=soulx-peagle
```

The wrapper maps `WANDB=1` to Speculators' `--logger wandb` option. Set
`WANDB_MODE=offline` or `WANDB_API_KEY=...` in the PJM file if the compute node
cannot authenticate interactively.

```bash
pjsub scripts/peagle/submit_peagle_online_h100.pjm
```

Use `LOGGER=tensorboard,wandb` in the PJM file if you want multiple Speculators
logger backends. `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE=offline`, and
`WANDB_API_KEY` are read by the `wandb` package from the environment.

The prepare stage uses a batched `datasets.map` path when the input dataset
already has `speech_tokens`. If `outputs/peagle_soulx_h100/preprocessed`
already contains `token_freq.pt` and `soulx_peagle_prepare_summary.json`, the
PJM script reuses it only after validating that the draft vocabulary maps back
to SoulX speech tokens. Set `FORCE_PREPARE=1` to rebuild.

SoulX has roughly 6561 speech tokens plus `semantic_token_end`, so the H100 PJM
defaults `DRAFT_VOCAB_SIZE=6562`. Do not use `8192` unless the validator also
passes. Speculators' vocabulary builder fills missing draft slots from low
target-token IDs and sorts the result, which can put text tokens into the draft
vocabulary when the requested draft vocab is larger than the speech-token
coverage. The wrapper now writes explicit `vocab_mapping/d2t.npy` and
`vocab_mapping/t2d.npy` from the validated `token_freq.pt` and passes them to
upstream `scripts/train.py`.

The PJM script has a shell-level guard for this now: if `DRAFT_VOCAB_SIZE=8192`
is still exported in your login/session environment, the job exits before
starting vLLM. The training wrapper has the same guard. Override it only for a
deliberate ablation with `ALLOW_OVERLARGE_DRAFT_VOCAB=1` or
`--allow-overlarge-draft-vocab`.

Important detail when inspecting checkpoints: upstream Speculators stores `d2t`
as an offset tensor. The effective target ID is `draft_index + d2t[draft_index]`,
not the raw `d2t` value alone.

The wrapper also patches upstream P-EAGLE training to preserve packed-sample
`position_ids` after COD downsampling. Without that patch, training and
validation use flattened packed-batch RoPE positions, while vLLM inference uses
normal per-request positions. Any checkpoint trained before this patch should be
discarded with `RESET_CHECKPOINTS=1`; the hidden-state cache and preprocessed
dataset can be reused. The PJM script records
`PEAGLE_TRAINING_CONTRACT_VERSION=2` in the work directory and automatically
moves older/no-marker checkpoints aside when
`AUTO_RESET_INCOMPATIBLE_CHECKPOINTS=1`.

Validate a prepared dataset or a trained checkpoint manually:

```bash
python scripts/peagle/validate_soulx_peagle_artifacts.py \
  --preprocessed-dir outputs/peagle_soulx_h100/preprocessed \
  --model-path runs/merged \
  --draft-vocab-size 6562 \
  --expected-num-depths 4 \
  --expected-num-layers 4 \
  --expected-draft-arch llama \
  --checkpoint outputs/peagle_soulx_h100/checkpoints/checkpoint_best
```

Inspect the runtime contract after training:

```bash
python scripts/peagle/inspect_peagle_runtime_contract.py \
  --model-path runs/merged \
  --checkpoint outputs/peagle_soulx_h100/checkpoints/checkpoint_best \
  --preprocessed-dir outputs/peagle_soulx_h100/preprocessed \
  --hidden-states-dir outputs/peagle_soulx_h100/hidden_states \
  --speculative-config exports/peagle/speculative_config.json \
  --expected-draft-vocab-size 6562 \
  --expected-num-depths 4 \
  --fail-on-error
```

If runtime agreement is far below validation agreement, first verify that the
checkpoint's training verifier and the deployed verifier are bit-identical for
the validation head:

```bash
python scripts/peagle/compare_peagle_verifier_weights.py \
  --checkpoint outputs/peagle_soulx_h100/checkpoints/checkpoint_best \
  --runtime-model runs/merged \
  --fail-on-mismatch
```

This checks the verifier path recorded in `checkpoint_best/config.json` against
the runtime model, including `lm_head.weight` and `model.norm.weight`.

If those weights match, compare a few cached training hidden-state files against
fresh hidden states from the live vLLM extractor:

```bash
python scripts/peagle/compare_peagle_hidden_state_cache.py \
  --preprocessed-dir outputs/peagle_soulx_h100/preprocessed \
  --hidden-states-dir outputs/peagle_soulx_h100/hidden_states \
  --endpoint http://127.0.0.1:8000/v1 \
  --num-samples 3 \
  --fail-on-mismatch
```

Run it while the same hidden-state vLLM server used for P-EAGLE training is up.
It requests the exact `input_ids` from the prepared dataset, loads the temporary
live `.safetensors` file, and compares it with `hs_<index>.safetensors`.

If you previously trained with `DRAFT_VOCAB_SIZE=8192`, use a fresh `WORK_DIR`
or set `RESET_CHECKPOINTS=1 FORCE_PREPARE=1 DRAFT_VOCAB_SIZE=6562` for the next
PJM submission. The PJM script also defaults
`AUTO_RESET_INCOMPATIBLE_CHECKPOINTS=1`: if the old checkpoint fails the vocab
validator, it is moved aside to `checkpoints.incompatible_*` and training starts
fresh with the validated 6562-token mapping.

The hidden-state vLLM server is launched with `--no-enable-chunked-prefill`
because Speculators' `ExampleHiddenStatesConnector` rejects chunked prefill in
vLLM `0.22.x`.

The PJM script also disables optional DeepGEMM FP8 kernel paths by default:
`VLLM_USE_DEEP_GEMM=0`, `VLLM_MOE_USE_DEEP_GEMM=0`,
`VLLM_USE_DEEP_GEMM_E8M0=0`, and `VLLM_DEEP_GEMM_WARMUP=skip`. This avoids
startup failures on H100 nodes where vLLM `0.22.x` detects DeepGEMM support but
the installed `deep_gemm` package is missing or too old.
Before training, the PJM script applies the same Speculators index patch as the
setup helper, so resubmitted jobs do not need a manual edit inside
`third_party/speculators`.

Prepare the dataset:

```bash
python scripts/peagle/train_soulx_peagle.py \
  --stage prepare \
  --dataset-path data/your_full_dataset \
  --model-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --work-dir outputs/peagle_soulx \
  --max-samples 5000 \
  --overwrite-preprocessed
```

Start hidden-state extraction in a vLLM env. This process stays running:

```bash
python scripts/peagle/train_soulx_peagle.py \
  --stage launch-vllm \
  --speculators-root third_party/speculators \
  --model-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --work-dir outputs/peagle_soulx \
  --vllm-arg=--data-parallel-size --vllm-arg=1 \
  --vllm-arg=--port --vllm-arg=8000
```

In a second shell, generate hidden states:

```bash
python scripts/peagle/train_soulx_peagle.py \
  --stage generate-hidden-states \
  --speculators-root third_party/speculators \
  --model-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --work-dir outputs/peagle_soulx \
  --endpoint http://localhost:8000/v1 \
  --max-samples 5000 \
  --validate-outputs
```

Then train and write the runtime config:

```bash
python scripts/peagle/train_soulx_peagle.py \
  --stage train \
  --speculators-root third_party/speculators \
  --model-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --work-dir outputs/peagle_soulx

python scripts/peagle/train_soulx_peagle.py \
  --stage write-config \
  --model-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --work-dir outputs/peagle_soulx \
  --num-layers 2 \
  --num-depths 2
```

The wrapper defaults to a conservative 3090-oriented run:
`--num-layers 2 --num-depths 2 --train-total-seq-len 1024
--draft-vocab-size 6562`. For H100/H200, try `--num-layers 4 --num-depths 4
--train-total-seq-len 2048` or higher.

Use SoulX speech-token sequences, not general chat text, for a meaningful
acceptance rate. The current MTP dataset columns (`text`, `speech_tokens`,
`lang`) are the source data, and `prepare_soulx_dataset.py` converts them into
the Speculators `input_ids` / `loss_mask` / `seq_len` schema.
