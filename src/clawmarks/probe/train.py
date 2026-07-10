"""
Launch one kohya_ss (sd-scripts) SDXL LoRA training run on the remote pod, parameterized for
the probe-then-commit search: one hyperparameter direction, at either probe length (260 steps,
one full cosine cycle, revised 2026-07-09 from the calibration check's 156 steps, which cut a
probe off mid-cycle) or full length (780 steps), then pulls the resulting checkpoint(s) back to
the local scratch dir.

Baseline (current-best, epoch 4) config being varied from: network_dim 32, network_alpha 16,
unet_lr 1e-4, text_encoder_lr 5e-5, min_snr_gamma 5, lr_scheduler cosine (3 cycles), clip_skip 2,
resolution 1024, train_batch_size 4, AdamW8bit, bf16, 10 epochs / ~780 steps total (originally
31 images at 10 repeats each, batch size 4: ceil(310/4)=78 steps/epoch x 10 = 780; the dataset now
splits into 30 images at 10 repeats + 1 down-weighted outlier at 3 repeats, ~76 steps/epoch, close
enough that max_train_steps is passed explicitly rather than derived from epoch count).

Usage:
  python3 notes/train_probe.py --name dim64 --max-train-steps 260 \
      --network-dim 64 --network-alpha 32 --seed 20260709
  python3 notes/train_probe.py --name dim64 --max-train-steps 780 \
      --network-dim 64 --network-alpha 32

Each run writes remote checkpoints to /workspace/output/<name>_<steps>/ and downloads every
saved epoch checkpoint into notes/probe_runs/<name>_<steps>/ locally.

--seed pins the training seed (weight init + batch shuffle order, via kohya's set_seed,
confirmed to cover both since the DataLoader's shuffle draws from the same global torch RNG,
with no separate generator). Round 1 onward uses paired seeds: control and every candidate
direction are each run once per seed in CANONICAL_SEEDS below, so replicate i of any direction
shares its training seed with replicate i of control. This cancels the seed-driven component of
score variance out of the paired delta (direction - control), rather than just averaging over
it, without collapsing to a single seed (which would give n=1 effective replication and false
confidence that a lucky seed is a real effect). Omit --seed only for exploratory/one-off runs
that won't be compared against a paired control.
"""

# Fixed seed list reused across every direction and control probe from round 1 onward, so that
# replicate i of any direction shares a training seed with replicate i of control (paired design,
# see --seed above). Do not reorder or reuse a subset: pairing depends on the same index
# mapping to the same seed everywhere.
CANONICAL_SEEDS = [20260709, 8675309, 271828, 141421, 314159, 161803, 57721, 30103]
import argparse
import json
import os
import re
import sys

import paramiko

from clawmarks.config import ROOT

KEY_PATH = f"{ROOT}/runpod-ssh/id_ed25519"

# Baseline hyperparameters (epoch-4 current-best config), overridden per-direction by CLI flags.
DEFAULTS = dict(
    network_dim=32,
    network_alpha=16,
    unet_lr=1e-4,
    text_encoder_lr=5e-5,
    min_snr_gamma=5,
    lr_scheduler="cosine",
    lr_scheduler_num_cycles=3,
    clip_skip=2,
    resolution=1024,
    train_batch_size=4,
)


def read_host_port(host_module):
    text = open(host_module).read()
    host = re.search(r'HOST = "(.*)"', text).group(1)
    port = int(re.search(r"PORT = (\d+)", text).group(1))
    return host, port


def ssh_client(pod):
    host_module = f"{ROOT}/rpssh.py" if pod == 1 else f"{ROOT}/rpssh{pod}.py"
    host, port = read_host_port(host_module)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    client.connect(host, port=port, username="root", pkey=pkey, timeout=20)
    client.get_transport().set_keepalive(30)
    return client


def run_cmd(client, cmd, timeout=None):
    print(f"+ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    code = stdout.channel.recv_exit_status()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print("STDERR:", err, file=sys.stderr)
    return code


def build_train_command(args, run_name, output_subdir, save_every_n_epochs):
    return " ".join([
        "source /workspace/venv/bin/activate &&",
        "cd /workspace/kohya_ss &&",
        "accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py",
        "--pretrained_model_name_or_path /workspace/models/illustrious_v0.1.safetensors",
        "--train_data_dir /workspace/training/img",
        "--caption_extension .txt",
        f"--output_dir /workspace/output/{output_subdir}",
        f"--output_name {run_name}",
        "--save_model_as safetensors",
        f"--save_every_n_epochs {save_every_n_epochs}",
        f"--max_train_steps {args.max_train_steps}",
        f"--resolution {args.resolution}",
        f"--train_batch_size {args.train_batch_size}",
        "--network_module networks.lora",
        f"--network_dim {args.network_dim}",
        f"--network_alpha {args.network_alpha}",
        f"--unet_lr {args.unet_lr}",
        f"--text_encoder_lr {args.text_encoder_lr}",
        f"--min_snr_gamma {args.min_snr_gamma}",
        f"--lr_scheduler {args.lr_scheduler}",
        (f"--lr_scheduler_num_cycles {args.lr_scheduler_num_cycles}" if args.lr_scheduler == "cosine" else ""),
        f"--clip_skip {args.clip_skip}",
        (f"--seed {args.seed}" if args.seed is not None else ""),
        "--optimizer_type AdamW8bit",
        "--mixed_precision bf16",
        "--save_precision bf16",
        "--xformers",
        "--cache_latents",
        "--gradient_checkpointing",
        f"> /workspace/output/{output_subdir}/train.log 2>&1",
    ])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="direction label, e.g. dim64, lr2e4, constlr")
    ap.add_argument("--max-train-steps", type=int, required=True, help="260 for probe length (one cosine cycle), 780 for full length")
    ap.add_argument("--network-dim", type=int, default=DEFAULTS["network_dim"])
    ap.add_argument("--network-alpha", type=int, default=DEFAULTS["network_alpha"])
    ap.add_argument("--unet-lr", type=float, default=DEFAULTS["unet_lr"])
    ap.add_argument("--text-encoder-lr", type=float, default=DEFAULTS["text_encoder_lr"])
    ap.add_argument("--min-snr-gamma", type=float, default=DEFAULTS["min_snr_gamma"])
    ap.add_argument("--lr-scheduler", default=DEFAULTS["lr_scheduler"], choices=["cosine", "constant"])
    ap.add_argument("--lr-scheduler-num-cycles", type=int, default=DEFAULTS["lr_scheduler_num_cycles"])
    ap.add_argument("--clip-skip", type=int, default=DEFAULTS["clip_skip"])
    ap.add_argument("--resolution", type=int, default=DEFAULTS["resolution"])
    ap.add_argument("--train-batch-size", type=int, default=DEFAULTS["train_batch_size"])
    ap.add_argument("--save-every-n-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="training seed (weight init + batch shuffle order). Omit for an uncontrolled/random seed.")
    ap.add_argument("--pod", type=int, default=1, choices=[1, 2], help="which pod's rpssh*.py to target")
    args = ap.parse_args(argv)

    output_subdir = f"{args.name}_{args.max_train_steps}"
    run_name = f"{args.name}_{args.max_train_steps}"

    client = ssh_client(args.pod)
    run_cmd(client, f"mkdir -p /workspace/output/{output_subdir}")
    cmd = build_train_command(args, run_name, output_subdir, args.save_every_n_epochs)
    code = run_cmd(client, cmd, timeout=3600)
    if code != 0:
        print(f"TRAINING FAILED (exit {code}), see remote log at /workspace/output/{output_subdir}/train.log")
        client.close()
        sys.exit(code)

    local_dir = f"{ROOT}/notes/probe_runs/{output_subdir}"
    os.makedirs(local_dir, exist_ok=True)
    sftp = client.open_sftp()
    for fname in sftp.listdir(f"/workspace/output/{output_subdir}"):
        if fname.endswith(".safetensors"):
            sftp.get(f"/workspace/output/{output_subdir}/{fname}", f"{local_dir}/{fname}")
            print(f"downloaded {fname}")
    sftp.get(f"/workspace/output/{output_subdir}/train.log", f"{local_dir}/train.log")
    sftp.close()
    client.close()
    print(f"DONE: {output_subdir} -> {local_dir}")


if __name__ == "__main__":
    main()
