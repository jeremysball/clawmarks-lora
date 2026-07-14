"""Tiny RunPod GraphQL client for balance checks, shared by curation_server.py's counterfactual
endpoint and search/run_manager.py's pre-launch safety check."""
import json
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"


def runpod_balance(api_key):
    req = urllib.request.Request(
        f"{GRAPHQL_URL}?api_key={api_key}",
        data=json.dumps({"query": "query { myself { clientBalance } }"}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    return res["data"]["myself"]["clientBalance"]
