"""
RunPod compute backend: pod bring-up, GraphQL control, SSH, and SFTP. Merged from
rp_bring_up.py + rp_bring_up2.py (the "2" suffix was only ever about running two pods
concurrently, not different logic) and rpget/rpsftp/rpssh, parameterized by pod_id instead
of duplicated.

Each SSH/SFTP operation looks up the pod's (host, port) in a module-level registry that
bring_up() populates. To target a pod that was brought up outside this module (e.g. a pod
left running from a previous session), call register_pod(pod_id, host, port) first.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from clawmarks.config import ROOT

API_KEY = os.environ["RUNPOD_API_KEY"]
CIVITAI_TOKEN = os.environ["CIVITAI_TOKEN"]
GRAPHQL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

KEY_PATH = str(ROOT / "runpod-ssh" / "id_ed25519")
PUBLIC_KEY_PATH = str(ROOT / "runpod-ssh" / "id_ed25519.pub")

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME_PREFIX = "clawmarks-training"

DEFAULT_GPU_PRIORITY = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000"]
DEFAULT_DATASET_ZIP = str(ROOT / "clawmarks-dataset.zip")
DEFAULT_DATASET_DIR = str(ROOT / "corrected_dataset_extract")
DEFAULT_REMOTE_SETUP_SCRIPT = str(ROOT / "notes" / "remote_setup.sh")
DEFAULT_POD_INDEX = 1

_known_pods = {}


def register_pod(pod_id, host, port):
    """Manually register a pod's (host, port) for SSH/SFTP operations. Use for pods that
    were brought up outside this module (e.g. a leftover pod from a previous session)."""
    _known_pods[pod_id] = (host, port)


def gql(query):
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    if "errors" in res:
        raise RuntimeError(res["errors"])
    return res["data"]


def get_balance() -> float:
    return gql("query { myself { clientBalance } }")["myself"]["clientBalance"]


def pause(pod_id):
    gql(f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) }}')
    print(f"paused pod {pod_id}")


def terminate(pod_id):
    gql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
    print(f"terminated pod {pod_id}")


def _stock_status(gpu_id):
    data = gql(f'query {{ gpuTypes(input: {{id: "{gpu_id}"}}) {{ lowestPrice(input: {{gpuCount: 1}}) {{ stockStatus uninterruptablePrice }} }} }}')
    lp = data["gpuTypes"][0]["lowestPrice"]
    return lp["stockStatus"], lp["uninterruptablePrice"]


def _pick_gpu(gpu_priority):
    for gpu in gpu_priority:
        status, price = _stock_status(gpu)
        print(f"  {gpu}: stock={status} price={price}")
        if status in ("High", "Medium"):
            return gpu
    raise RuntimeError("no GPU in the priority list has usable stock right now")


def _create_pod(gpu_id, pod_name):
    public_key = open(PUBLIC_KEY_PATH).read().strip()
    mutation = f'''
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: SECURE
        gpuCount: 1
        volumeInGb: 60
        containerDiskInGb: 60
        minVcpuCount: 4
        minMemoryInGb: 15
        gpuTypeId: "{gpu_id}"
        name: "{pod_name}"
        imageName: "{IMAGE}"
        ports: "22/tcp"
        volumeMountPath: "/workspace"
        env: [{{ key: "PUBLIC_KEY", value: "{public_key}" }}]
      }}) {{
        id
      }}
    }}'''
    data = gql(mutation)
    return data["podFindAndDeployOnDemand"]["id"]


def _wait_for_ssh(pod_id, timeout=600):
    print(f"waiting for pod {pod_id} to boot and expose SSH...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = gql(f'''query {{
          pod(input: {{podId: "{pod_id}"}}) {{
            id desiredStatus
            runtime {{ ports {{ ip isIpPublic privatePort publicPort type }} }}
          }}
        }}''')
        pod = data["pod"]
        runtime = pod.get("runtime")
        if runtime and runtime.get("ports"):
            for p in runtime["ports"]:
                if p["privatePort"] == 22 and p["isIpPublic"]:
                    print(f"  SSH ready: {p['ip']}:{p['publicPort']}")
                    return p["ip"], p["publicPort"]
        elapsed = int(time.time() - t0)
        print(f"  [{elapsed}s] status={pod['desiredStatus']} runtime={'up' if runtime else 'not yet'}")
        time.sleep(10)
    raise TimeoutError("pod never exposed a public SSH port in time")


def _ssh_client(host, port):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    for attempt in range(10):
        try:
            client.connect(host, port=port, username="root", pkey=pkey, timeout=20)
            return client
        except Exception as e:
            print(f"  ssh connect attempt {attempt+1} failed ({e}), retrying...")
            time.sleep(10)
    raise RuntimeError("could not establish SSH after pod reported ready")


def _run_cmd(client, cmd, timeout=1800):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    code = stdout.channel.recv_exit_status()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print("STDERR:", err, file=sys.stderr)
    return code


def _upload_dataset_and_setup(client, dataset_zip, dataset_dir, setup_script):
    import zipfile
    sftp = client.open_sftp()
    sftp.put(setup_script, "/workspace/remote_setup.sh")

    if not os.path.exists(dataset_zip):
        with zipfile.ZipFile(dataset_zip, "w") as z:
            for f in os.listdir(dataset_dir):
                z.write(os.path.join(dataset_dir, f), f)
        print(f"zipped dataset -> {dataset_zip}")
    sftp.put(dataset_zip, "/workspace/clawmarks-dataset.zip")
    sftp.close()
    print("uploaded dataset + setup script")

    _run_cmd(client, f"chmod +x /workspace/remote_setup.sh && CIVITAI_TOKEN={CIVITAI_TOKEN} /workspace/remote_setup.sh")


def bring_up(gpu_priority=None, pod_index=None, dataset_zip=None, dataset_dir=None, setup_script=None) -> dict:
    """Deploy a new on-demand pod. Returns {"pod_id", "host", "port"} and registers the
    pod in the module-level registry so subsequent ssh/get/put calls can target it.
    pod_index (1 or 2) chooses the pod name suffix so two pods can run concurrently."""
    priority = gpu_priority or DEFAULT_GPU_PRIORITY
    idx = pod_index or DEFAULT_POD_INDEX
    pod_name = POD_NAME_PREFIX if idx == 1 else f"{POD_NAME_PREFIX}-{idx}"
    zip_path = dataset_zip or DEFAULT_DATASET_ZIP
    dir_path = dataset_dir or DEFAULT_DATASET_DIR
    setup_path = setup_script or DEFAULT_REMOTE_SETUP_SCRIPT

    print("checking GPU availability...")
    gpu_id = _pick_gpu(priority)
    print(f"deploying on {gpu_id}")
    pod_id = _create_pod(gpu_id, pod_name)
    print(f"pod created: {pod_id}")
    host, port = _wait_for_ssh(pod_id)
    register_pod(pod_id, host, port)
    client = _ssh_client(host, port)
    _upload_dataset_and_setup(client, zip_path, dir_path, setup_path)
    client.close()
    print(f"\nDONE. Pod {pod_id} at {host}:{port} ready for training.")
    return {"pod_id": pod_id, "host": host, "port": port}


def _resolve(pod_id):
    if pod_id not in _known_pods:
        raise RuntimeError(
            f"pod {pod_id!r} is not registered with compute/runpod.py. "
            f"Call bring_up() first, or register_pod(pod_id, host, port) if it was brought "
            f"up out-of-band."
        )
    return _known_pods[pod_id]


def ssh(pod_id, command, timeout=None) -> subprocess.CompletedProcess:
    """Run a single command on the pod over SSH. Returns a CompletedProcess-like result with
    stdout/stderr/returncode populated, matching subprocess.run conventions."""
    import paramiko
    host, port = _resolve(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    client.connect(host, port=port, username="root", pkey=pkey, timeout=20)
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    client.close()
    print(out)
    if err:
        print("STDERR:", err, file=sys.stderr)
    print(f"EXIT:{code}")
    return subprocess.CompletedProcess(args=command, returncode=code, stdout=out, stderr=err)


def get(pod_id, remote_path, local_path) -> None:
    import paramiko
    host, port = _resolve(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    client.connect(host, port=port, username="root", pkey=pkey, timeout=20)
    sftp = client.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    client.close()
    print("downloaded", remote_path, "->", local_path)


def put(pod_id, local_path, remote_path) -> None:
    import paramiko
    host, port = _resolve(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    client.connect(host, port=port, username="root", pkey=pkey, timeout=20)
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    client.close()
    print("uploaded", local_path, "->", remote_path)
