import os

# clawmarks.compute.comfyui, clawmarks.compute.runpod, and clawmarks.probe.sweep read their
# secrets at module scope (API_KEY = os.environ["RUNPOD_API_KEY"]), so merely importing them
# raises KeyError where no secret is set. That fail-fast is right for a script run against real
# RunPod billing, but it means a fresh clone or a CI runner cannot even collect the suite. Seed
# placeholders before collection imports anything. setdefault, so a real .envrc still wins and no
# test can silently run against a fake key when a real one is loaded.
os.environ.setdefault("RUNPOD_API_KEY", "test-placeholder-not-a-real-key")
os.environ.setdefault("CIVITAI_TOKEN", "test-placeholder-not-a-real-token")
