#!/usr/bin/env python3
"""Run the SoulX P-EAGLE training stages around upstream Speculators scripts."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_int_list(values: list[str] | None) -> list[str]:
    return [] if values is None else [str(int(v)) for v in values]


def script_path(speculators_root: Path, name: str, *, require: bool = True) -> Path:
    path = speculators_root / "scripts" / name
    if require and not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Clone https://github.com/vllm-project/speculators "
            "and pass --speculators-root /path/to/speculators."
        )
    return path


def run(
    cmd: list[str],
    *,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(printable)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def common_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_dir = Path(args.work_dir)
    return {
        "work_dir": work_dir,
        "preprocessed": work_dir / "preprocessed",
        "hidden_states": work_dir / "hidden_states",
        "checkpoints": work_dir / "checkpoints",
        "speculative_config": Path(args.speculative_config_output),
    }


def speculators_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    spec_root_path = Path(args.speculators_root).resolve()
    spec_paths = [
        str(spec_root_path / "src"),
        str(spec_root_path),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(spec_paths + ([existing] if existing else []))
    return env


def patch_speculators_checkout(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "peagle" / "patch_speculators_checkout.py"),
            "--speculators-root",
            args.speculators_root,
        ],
        cwd=ROOT,
        check=True,
    )


def logger_arg(args: argparse.Namespace) -> str:
    loggers = [item.strip() for item in args.logger.split(",") if item.strip()]
    if args.wandb and "wandb" not in loggers:
        loggers.append("wandb")
    return ",".join(loggers)


def validate_draft_vocab_size(args: argparse.Namespace) -> None:
    if args.allow_overlarge_draft_vocab:
        return
    max_safe = args.speech_vocab_size + 1 + args.max_non_speech_draft_tokens
    if args.draft_vocab_size > max_safe:
        raise ValueError(
            f"--draft-vocab-size {args.draft_vocab_size} is too large for "
            f"SoulX speech tokens. Use 6562 unless you know the draft vocab "
            f"should include non-speech target IDs. Safe maximum with current "
            f"validation settings is {max_safe}."
        )


def validate_preprocessed(args: argparse.Namespace, *, write_mapping: bool) -> None:
    if args.skip_vocab_validation:
        return
    paths = common_paths(args)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "peagle" / "validate_soulx_peagle_artifacts.py"),
        "--preprocessed-dir",
        str(paths["preprocessed"]),
        "--model-path",
        args.model_path,
        "--draft-vocab-size",
        str(args.draft_vocab_size),
        "--speech-vocab-size",
        str(args.speech_vocab_size),
        "--min-speech-draft-tokens",
        str(args.min_speech_draft_tokens),
        "--max-non-speech-draft-tokens",
        str(args.max_non_speech_draft_tokens),
    ]
    if write_mapping:
        cmd += [
            "--write-vocab-mapping",
            "--mapping-output-dir",
            str(paths["work_dir"] / "vocab_mapping"),
        ]
    if args.expected_prepare_mode:
        cmd += ["--expected-prepare-mode", args.expected_prepare_mode]
    run(cmd, dry_run=args.dry_run)


def validate_checkpoint(args: argparse.Namespace) -> None:
    if args.skip_vocab_validation:
        return
    paths = common_paths(args)
    checkpoint = paths["checkpoints"] / "checkpoint_best"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "peagle" / "validate_soulx_peagle_artifacts.py"),
        "--preprocessed-dir",
        str(paths["preprocessed"]),
        "--model-path",
        args.model_path,
        "--draft-vocab-size",
        str(args.draft_vocab_size),
        "--speech-vocab-size",
        str(args.speech_vocab_size),
        "--min-speech-draft-tokens",
        str(args.min_speech_draft_tokens),
        "--max-non-speech-draft-tokens",
        str(args.max_non_speech_draft_tokens),
        "--checkpoint",
        str(checkpoint),
        "--expected-num-depths",
        str(args.num_depths),
        "--expected-num-layers",
        str(args.num_layers),
        "--expected-draft-arch",
        args.draft_arch,
    ]
    target_layer_ids = parse_int_list(args.target_layer_ids)
    if target_layer_ids:
        cmd += ["--expected-target-layer-ids", *target_layer_ids]
    if args.expected_prepare_mode:
        cmd += ["--expected-prepare-mode", args.expected_prepare_mode]
    run(cmd, dry_run=args.dry_run)


def prepare(args: argparse.Namespace) -> None:
    paths = common_paths(args)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "peagle" / "prepare_soulx_dataset.py"),
        "--dataset-path",
        args.dataset_path,
        "--model-path",
        args.model_path,
        "--output-dir",
        str(paths["preprocessed"]),
        "--seq-length",
        str(args.prepare_seq_length),
        "--min-speech-tokens",
        str(args.min_speech_tokens),
        "--max-speech-tokens",
        str(args.max_speech_tokens),
        "--seed",
        str(args.seed),
        "--map-batch-size",
        str(args.prepare_map_batch_size),
        "--num-proc",
        str(args.prepare_num_proc),
    ]
    if args.dataset_split:
        cmd += ["--split", args.dataset_split]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.shuffle_prepare:
        cmd.append("--shuffle")
    if args.skip_dialect_prefix:
        cmd.append("--skip-dialect-prefix")
    if args.no_eos_loss:
        cmd.append("--no-eos-loss")
    if args.allow_audio_tokenize_fallback:
        cmd.append("--allow-audio-tokenize-fallback")
    if args.legacy_prepare_loop:
        cmd.append("--legacy-loop")
    if args.overwrite_preprocessed:
        cmd.append("--overwrite")
    run(cmd, dry_run=args.dry_run)


def prepare_generated(args: argparse.Namespace) -> None:
    paths = common_paths(args)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "peagle" / "prepare_soulx_generated_dataset.py"),
        "--dataset-path",
        args.dataset_path,
        "--model-path",
        args.model_path,
        "--output-dir",
        str(paths["preprocessed"]),
        "--endpoint",
        args.endpoint,
        "--seq-length",
        str(args.prepare_seq_length),
        "--min-speech-tokens",
        str(args.min_speech_tokens),
        "--max-speech-tokens",
        str(args.max_speech_tokens),
        "--speech-vocab-size",
        str(args.speech_vocab_size),
        "--seed",
        str(args.seed),
        "--concurrency",
        str(args.generation_concurrency),
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
        "--temperature",
        str(args.generation_temperature),
        "--top-k",
        str(args.generation_top_k),
        "--top-p",
        str(args.generation_top_p),
        "--repetition-penalty",
        str(args.generation_repetition_penalty),
    ]
    if args.dataset_split:
        cmd += ["--split", args.dataset_split]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.generation_max_tokens is not None:
        cmd += ["--generation-max-tokens", str(args.generation_max_tokens)]
    if args.shuffle_prepare:
        cmd.append("--shuffle")
    if args.skip_dialect_prefix:
        cmd.append("--skip-dialect-prefix")
    if args.no_eos_loss:
        cmd.append("--no-eos-loss")
    if args.overwrite_preprocessed:
        cmd.append("--overwrite")
    if args.resume_generated_prepare:
        cmd.append("--resume")
    if args.multi_speaker:
        cmd.append("--multi-speaker")
        cmd += ["--num-speakers", str(args.num_speakers)]
        if args.num_dialogues is not None:
            cmd += ["--num-dialogues", str(args.num_dialogues)]
        if args.lang_mix:
            cmd += ["--lang-mix", args.lang_mix]
    run(cmd, dry_run=args.dry_run)


def launch_vllm(args: argparse.Namespace) -> None:
    paths = common_paths(args)
    spec_root = Path(args.speculators_root)
    cmd = [
        args.vllm_python,
        str(script_path(spec_root, "launch_vllm.py", require=not args.dry_run)),
        args.model_path,
        "--hidden-states-path",
        str(paths["hidden_states"]),
    ]
    target_layer_ids = parse_int_list(args.target_layer_ids)
    if target_layer_ids:
        cmd += ["--target-layer-ids", *target_layer_ids]
    cmd += ["--", *args.vllm_arg]
    run(cmd, dry_run=args.dry_run, env=speculators_env(args))


def generate_hidden_states(args: argparse.Namespace) -> None:
    paths = common_paths(args)
    spec_root = Path(args.speculators_root)
    cmd = [
        args.speculators_python,
        str(
            script_path(
                spec_root,
                "data_generation_offline.py",
                require=not args.dry_run,
            )
        ),
        "--model",
        args.model_path,
        "--preprocessed-data",
        str(paths["preprocessed"]),
        "--endpoint",
        args.endpoint,
        "--output",
        str(paths["hidden_states"]),
        "--concurrency",
        str(args.concurrency),
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.validate_outputs:
        cmd.append("--validate-outputs")
    if args.fail_on_error:
        cmd.append("--fail-on-error")
    run(cmd, dry_run=args.dry_run, env=speculators_env(args))


def train(args: argparse.Namespace) -> None:
    patch_speculators_checkout(args)
    paths = common_paths(args)
    validate_draft_vocab_size(args)
    validate_preprocessed(args, write_mapping=args.explicit_vocab_mapping)
    spec_root = Path(args.speculators_root)
    train_script = script_path(spec_root, "train.py", require=not args.dry_run)
    base_cmd = [
        str(train_script),
        "--verifier-name-or-path",
        args.model_path,
        "--data-path",
        str(paths["preprocessed"]),
        "--hidden-states-path",
        str(paths["hidden_states"]),
        "--save-path",
        str(paths["checkpoints"]),
        "--speculator-type",
        "peagle",
        "--num-layers",
        str(args.num_layers),
        "--draft-arch",
        args.draft_arch,
        "--num-depths",
        str(args.num_depths),
        "--down-sample-ratio",
        str(args.down_sample_ratio),
        "--down-sample-ratio-min",
        str(args.down_sample_ratio_min),
        "--scheduler-type",
        args.scheduler_type,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--total-seq-len",
        str(args.train_total_seq_len),
        "--draft-vocab-size",
        str(args.draft_vocab_size),
        "--token-freq-path",
        str(paths["preprocessed"] / "token_freq.pt"),
        "--on-missing",
        args.on_missing,
        "--hidden-states-dtype",
        args.hidden_states_dtype,
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--seed",
        str(args.seed),
    ]
    if args.explicit_vocab_mapping:
        base_cmd += [
            "--d2t-path",
            str(paths["work_dir"] / "vocab_mapping" / "d2t.npy"),
            "--t2d-path",
            str(paths["work_dir"] / "vocab_mapping" / "t2d.npy"),
        ]
    if args.on_missing == "generate":
        base_cmd += [
            "--vllm-endpoint",
            args.endpoint,
            "--on-generate",
            args.on_generate,
            "--request-timeout",
            str(args.request_timeout),
            "--max-retries",
            str(args.max_retries),
        ]
    target_layer_ids = parse_int_list(args.target_layer_ids)
    if target_layer_ids:
        base_cmd += ["--target-layer-ids", *target_layer_ids]
    base_cmd.append(
        "--norm-before-residual"
        if args.norm_before_residual
        else "--no-norm-before-residual"
    )
    if args.save_best:
        base_cmd.append("--save-best")
    logger = logger_arg(args)
    if logger:
        base_cmd += ["--logger", logger]
    if args.log_dir:
        base_cmd += ["--log-dir", args.log_dir]
    if args.run_name:
        base_cmd += ["--run-name", args.run_name]
    for extra in args.extra_train_arg:
        base_cmd.append(extra)

    if args.nproc_per_node > 1:
        cmd = [
            args.speculators_python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.nproc_per_node),
            *base_cmd,
        ]
    else:
        cmd = [args.speculators_python, *base_cmd]
    run(cmd, dry_run=args.dry_run, env=speculators_env(args))


def write_config(args: argparse.Namespace) -> None:
    validate_draft_vocab_size(args)
    validate_checkpoint(args)
    paths = common_paths(args)
    checkpoint = paths["checkpoints"] / "checkpoint_best"
    num_speculative_tokens = args.num_speculative_tokens or args.num_depths
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "peagle" / "write_speculative_config.py"),
        "--speculator-model",
        str(checkpoint),
        "--num-speculative-tokens",
        str(num_speculative_tokens),
        "--method",
        args.speculative_method,
        "--output",
        str(paths["speculative_config"]),
    ]
    if args.parallel_drafting:
        cmd.append("--parallel-drafting")
    else:
        cmd.append("--no-parallel-drafting")
    run(cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "prepare-generated",
            "launch-vllm",
            "generate-hidden-states",
            "train",
            "write-config",
            "offline",
            "online",
        ],
        required=True,
        help=(
            "'offline' runs prepare, generate-hidden-states, train, and write-config. "
            "'online' runs prepare, train with on-demand hidden-state generation, "
            "and write-config. Start launch-vllm separately for both online train "
            "and generate-hidden-states."
        ),
    )
    parser.add_argument("--dataset-path", default="data/your_full_dataset")
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--model-path", default="pretrained_models/SoulX-Podcast-1.7B-dialect")
    parser.add_argument("--work-dir", default="outputs/peagle_soulx")
    parser.add_argument("--speculators-root", default=os.getenv("SPECULATORS_ROOT", "third_party/speculators"))
    parser.add_argument("--speculators-python", default=sys.executable)
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--speculative-config-output", default="exports/peagle/speculative_config.json")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prepare-seq-length", type=int, default=2048)
    parser.add_argument("--min-speech-tokens", type=int, default=8)
    parser.add_argument("--max-speech-tokens", type=int, default=750)
    parser.add_argument("--prepare-map-batch-size", type=int, default=2000)
    parser.add_argument("--prepare-num-proc", type=int, default=1)
    parser.add_argument("--generation-concurrency", type=int, default=32)
    parser.add_argument("--generation-temperature", type=float, default=0.6)
    parser.add_argument("--generation-top-k", type=int, default=100)
    parser.add_argument("--generation-top-p", type=float, default=0.9)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.25)
    parser.add_argument("--generation-max-tokens", type=int, default=None)
    parser.add_argument("--resume-generated-prepare", action="store_true")
    parser.add_argument(
        "--multi-speaker",
        action="store_true",
        help="Forward --multi-speaker to prepare_soulx_generated_dataset.py.",
    )
    parser.add_argument(
        "--num-dialogues",
        type=int,
        default=None,
        help="Number of multi-speaker dialogues to generate (only with --multi-speaker).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speaker prompts per dialogue (only with --multi-speaker).",
    )
    parser.add_argument(
        "--lang-mix",
        type=str,
        default="",
        help="Language weights for multi-speaker mode, e.g. 'yue:1,zh:1,en:1'.",
    )
    parser.add_argument("--expected-prepare-mode", default=None)
    parser.add_argument("--legacy-prepare-loop", action="store_true")
    parser.add_argument("--shuffle-prepare", action="store_true")
    parser.add_argument("--skip-dialect-prefix", action="store_true")
    parser.add_argument("--no-eos-loss", action="store_true")
    parser.add_argument("--allow-audio-tokenize-fallback", action="store_true")
    parser.add_argument("--overwrite-preprocessed", action="store_true")

    parser.add_argument("--target-layer-ids", nargs="+", default=None)
    parser.add_argument("--vllm-arg", action="append", default=[], help="Extra arg passed after launch_vllm.py --")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--validate-outputs", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)

    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--draft-arch",
        default="llama",
        help=(
            "Upstream Speculators draft decoder architecture. Use 'llama' for "
            "current vLLM P-EAGLE inference compatibility."
        ),
    )
    parser.add_argument("--num-depths", type=int, default=2)
    parser.add_argument("--down-sample-ratio", type=float, default=0.7)
    parser.add_argument("--down-sample-ratio-min", type=float, default=0.2)
    parser.add_argument(
        "--norm-before-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--scheduler-type", default="cosine")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--train-total-seq-len", type=int, default=1024)
    parser.add_argument("--draft-vocab-size", type=int, default=6562)
    parser.add_argument("--speech-vocab-size", type=int, default=6561)
    parser.add_argument("--min-speech-draft-tokens", type=int, default=6000)
    parser.add_argument("--max-non-speech-draft-tokens", type=int, default=16)
    parser.add_argument("--skip-vocab-validation", action="store_true")
    parser.add_argument("--allow-overlarge-draft-vocab", action="store_true")
    parser.add_argument(
        "--explicit-vocab-mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build and pass explicit d2t/t2d .npy files from SoulX token_freq.pt. "
            "Disable only for debugging upstream Speculators defaults."
        ),
    )
    parser.add_argument(
        "--on-missing",
        choices=["generate", "skip", "warn", "raise"],
        default="raise",
    )
    parser.add_argument(
        "--on-generate",
        choices=["delete", "cache"],
        default="delete",
        help=(
            "Speculators behavior for hidden states generated during training. "
            "'delete' is pure online training; 'cache' keeps first-epoch states "
            "for reuse."
        ),
    )
    parser.add_argument("--hidden-states-dtype", default="bfloat16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging via upstream Speculators --logger wandb.",
    )
    parser.add_argument(
        "--logger",
        default="",
        help=(
            "Upstream Speculators logger backend(s), e.g. 'wandb' or "
            "'tensorboard,wandb'. --wandb appends wandb to this list."
        ),
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--extra-train-arg", action="append", default=[])
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument(
        "--speculative-method",
        default="eagle3",
        help="vLLM speculative_config method string. Use 'eagle3' for P-EAGLE.",
    )
    parser.add_argument(
        "--parallel-drafting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write parallel_drafting=true in the vLLM speculative_config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "prepare-generated":
        prepare_generated(args)
    elif args.stage == "launch-vllm":
        launch_vllm(args)
    elif args.stage == "generate-hidden-states":
        generate_hidden_states(args)
    elif args.stage == "train":
        train(args)
    elif args.stage == "write-config":
        write_config(args)
    elif args.stage == "offline":
        prepare(args)
        generate_hidden_states(args)
        train(args)
        write_config(args)
    elif args.stage == "online":
        args.on_missing = "generate"
        prepare(args)
        train(args)
        write_config(args)


if __name__ == "__main__":
    main()
