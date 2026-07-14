"""Tiny RunPod GraphQL client for balance checks, shared by curation_server.py's counterfactual
endpoint and search/run_manager.py's pre-launch safety check."""
import json
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"


def runpod_balance(api_key):
    # Authorization header, not a ?api_key= query param: a query string can end up in server
    # access logs or proxy history in a way a header doesn't.
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": "query { myself { clientBalance } }"}).encode(),
        headers={
            "Content-Type": "application/json", "User-Agent": "curl/8.0",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    return res["data"]["myself"]["clientBalance"]
