"""
Pushes batch inference results directly into Log Analytics
using the HTTP Data Collector API — populates Grafana immediately.
"""

import json, hashlib, hmac, base64, datetime, requests
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from io import BytesIO

WORKSPACE_ID  = "8ba01f23-8060-4af6-8431-6975a7b9d8cf"
SHARED_KEY    = "DlBgvrNfUeOst3ruG175jt26Oyko+AaXy2xmBKI1/gbflTp/w7CAEvALCVviNB/W+UvU04b6H8HdG2EcQhYz8A=="
LOG_TYPE      = "TrafficInference"   # becomes TrafficInference_CL in Log Analytics

STORAGE_ACCOUNT = "trafficproddlake"
GOLD_CONTAINER  = "gold-serving"


def build_signature(workspace_id, shared_key, date, content_length, method, content_type, resource):
    x_headers = f"x-ms-date:{date}"
    string_to_hash = f"{method}\n{content_length}\n{content_type}\n{x_headers}\n{resource}"
    bytes_to_hash = string_to_hash.encode("utf-8")
    decoded_key   = base64.b64decode(shared_key)
    encoded_hash  = base64.b64encode(
        hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
    ).decode("utf-8")
    return f"SharedKey {workspace_id}:{encoded_hash}"


def post_logs(records: list[dict]) -> int:
    body         = json.dumps(records).encode("utf-8")
    rfc1123date  = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    content_len  = len(body)
    resource     = "/api/logs"
    signature    = build_signature(
        WORKSPACE_ID, SHARED_KEY, rfc1123date, content_len,
        "POST", "application/json", resource
    )
    uri = f"https://{WORKSPACE_ID}.ods.opinsights.azure.com{resource}?api-version=2016-04-01"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": signature,
        "Log-Type":      LOG_TYPE,
        "x-ms-date":     rfc1123date,
    }
    r = requests.post(uri, data=body, headers=headers)
    return r.status_code


def load_gold_results() -> list[dict]:
    credential   = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )
    container = blob_service.get_container_client(GOLD_CONTAINER)
    blobs = sorted(b.name for b in container.list_blobs(name_starts_with="coco128/batch_results/")
                   if b.name.endswith("full_results.json"))
    if not blobs:
        print("No Gold results found")
        return []

    latest = blobs[-1]
    print(f"Loading: {latest}")
    data = blob_service.get_blob_client(GOLD_CONTAINER, latest).download_blob().readall()
    return json.loads(data)["frames"]


def main():
    print("Pushing batch results to Log Analytics...")
    frames = load_gold_results()
    if not frames:
        return

    # Build records in the format Grafana queries expect
    records = []
    now = datetime.datetime.utcnow()
    for i, f in enumerate(frames):
        # Spread timestamps over the last hour so charts show a trend line
        ts = now - datetime.timedelta(seconds=(len(frames) - i) * 27)
        for det in (f["detections"] or [{}]):
            records.append({
                "TimeGenerated":  ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "frame":          f["frame_index"],
                "vehicles":       f["vehicles"],
                "anomaly_score":  f["anomaly_score"] or 0,
                "accident":       f["accident"],
                "class":          det.get("class", ""),
                "confidence":     det.get("confidence", 0),
                "LogEntry":       json.dumps({
                    "frame":         f["frame_index"],
                    "vehicles":      f["vehicles"],
                    "anomaly_score": f["anomaly_score"],
                    "accident":      f["accident"],
                    "class":         det.get("class", ""),
                    **({"ACCIDENT DETECTED": True} if f["accident"] else {}),
                }),
            })
        if not f["detections"]:
            records.append({
                "TimeGenerated":  ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "frame":          f["frame_index"],
                "vehicles":       0,
                "anomaly_score":  f["anomaly_score"] or 0,
                "accident":       f["accident"],
                "class":          "",
                "confidence":     0,
                "LogEntry":       json.dumps({
                    "frame":         f["frame_index"],
                    "vehicles":      0,
                    "anomaly_score": f["anomaly_score"],
                    "accident":      f["accident"],
                    **({"ACCIDENT DETECTED": True} if f["accident"] else {}),
                }),
            })

    # Post in batches of 100
    for i in range(0, len(records), 100):
        batch  = records[i:i+100]
        status = post_logs(batch)
        print(f"  Batch {i//100 + 1}: {len(batch)} records → HTTP {status}")

    print(f"\nDone. {len(records)} records pushed to TrafficInference_CL")
    print("Allow 3-5 minutes for Log Analytics to index, then refresh Grafana.")


if __name__ == "__main__":
    main()
