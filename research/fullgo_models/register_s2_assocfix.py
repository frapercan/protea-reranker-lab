"""Register the S2-assocfix trio as RerankerModel rows via the platform's
import-by-reference endpoint (the mechanism the dossier describes).

For each category: upload the booster to MinIO under rerankers/<name>/model.txt,
then POST /reranker-models/import-by-reference with a run.json that pins
feature_schema_sha = 7fcecf26aa0a (the live-pipeline sha the champion S2 trio
used, so the SchemaShaMismatch guard accepts the drop-in swap) and a spec.yaml
whose training.cell = the category. force=True overwrites any stale row of the
same name.

Names: native-{nk,lk,pk}-s2-assocfix. Linked to dataset 46be427a.
Prints the three new RerankerModel ids (consumed by the TEST eval payload).
"""
from __future__ import annotations

import json
import sys

import requests
from minio import Minio

API = "http://localhost:8000"
APIKEY = open("/tmp/int8_apikey.txt").read().strip()
HDR = {"Authorization": f"ApiKey {APIKEY}"}
BUCKET = "protea"
BDIR = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_s2_assocfix"
DATASET_ID = "46be427a-0a9a-4cbd-b725-afdf9488cabc"
FEATURE_SCHEMA_SHA = "7fcecf26aa0a"  # live-pipeline sha (champion S2 trio)
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

CATS = ("nk", "lk", "pk")


def main() -> int:
    ids: dict[str, str] = {}
    for cat in CATS:
        name = f"native-{cat}-s2-assocfix"
        local = f"{BDIR}/ensemble_gbm_{cat.upper()}.txt"
        key = f"rerankers/{name}/model.txt"
        with open(local, "rb") as fh:
            data = fh.read()
        CLIENT.put_object(BUCKET, key, __import__("io").BytesIO(data), length=len(data))
        artifact_uri = f"s3://{BUCKET}/{key}"

        run = {
            "run_id": name,
            "metrics": {},
            "feature_importance": {},
            "features": {"feature_schema_sha": FEATURE_SCHEMA_SHA},
            "dataset": {"name": "fullgo-union-SELECT-160-220-227-c99db18-assoc"},
        }
        spec_yaml = f"name: {name}\ntraining:\n  cell: {cat}\n"
        body = {
            "artifact_uri": artifact_uri,
            "spec_yaml": spec_yaml,
            "run": run,
            "name": name,
            "dataset_id": DATASET_ID,
            "external_source": "beat-lafa-1@s2-assocfix-retrain",
            "prediction_set_id": "f377adae-047a-4e52-9e79-b43d5e722a70",
            "evaluation_set_id": "34a634a8-5739-4b04-923c-26db9eaab21e",
            "force": True,
        }
        r = requests.post(f"{API}/reranker-models/import-by-reference", headers=HDR,
                          json=body, timeout=60)
        if r.status_code not in (200, 201):
            print(f"{cat}: REGISTER FAILED {r.status_code} {r.text[:300]}", flush=True)
            return 1
        rid = r.json()["id"]
        ids[cat] = rid
        print(f"{cat}: registered {name} -> {rid} (artifact {artifact_uri})", flush=True)

    json.dump(ids, open(f"{BDIR}/reranker_ids.json", "w"), indent=2)
    print("RERANKER_IDS " + json.dumps(ids), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
