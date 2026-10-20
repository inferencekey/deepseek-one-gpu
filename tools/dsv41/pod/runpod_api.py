#!/usr/bin/env python3
"""RunPod REST for tools/dsv41/pod/devgpu.sh, python3 standard library only.

  runpod_api.py price  --gpu G --cloud C [--cuda-versions V]                          G's list price per hour in C now
  runpod_api.py body   --gpu G --cloud C [--dc DC] --image I --disk GB [--min-ram GB] [--network-volume ID]
                       [--cuda-versions V]                                            the create body (no key, no request)
  runpod_api.py create --gpu G --cloud C [--dc DC] --image I --disk GB [--min-ram GB] [--network-volume ID]
                       [--cuda-versions V]                                            creates one pod; prints its JSON
  runpod_api.py get ID                                                                 status, publicIp, ssh port, cost, RAM
  runpod_api.py delete ID                                                              deletes it (a missing pod is deleted)

The key is RUNPOD_API_KEY (your RunPod API key, from the environment only). Every request sends a
User-Agent (Cloudflare answers 403 "error code: 1010" without one). The SSH port is portMappings["22"] on publicIp,
not runtime.ports, which stays empty on many hosts. create exits 3 when RunPod has no such machine free (the caller
tries the next GPU or data center), 1 on any other error; the body asks for hosts whose driver runs one of the CUDA
versions --cuda-versions lists (comma separated, default 12.8,12.9,13.0: the cu128 flavor, driver >= 570.26; 13.0
for the cu130 flavor's sm_120 GPUs, driver >= 580.65: POD_CUDA_VERSIONS in tools/dsv41/flavors.sh) and no volume
of its own (the payload is a few MB). --min-ram GB (> 0)
asks for hosts with at least that much RAM for the pod's one GPU (REST v1's minRAMPerGPU); create and get print the
pod's memoryInGb, which devgpu.sh checks against it. --network-volume ID (tools/dsv41/pod/e07.sh's
NETWORK_VOLUME, a volume holding the model) mounts that RunPod network volume at /workspace (networkVolumeId, volumeMountPath); a
volume lives in one data center, so the create must name it (--dc).

price reads REST v2's GPU catalog (GET /v2/catalog/gpus with availability, the pod product, one GPU, the same CUDA
versions; REST v1 has no price list), so devgpu.sh refuses a dear GPU before creating anything: it prints
{"gpu", "cloud", "price", "availability"} and exits 3 when the cloud does not offer the GPU or has none available
now, 1 on any other error. The list price is per GPU and excludes the container disk; create's costPerHr includes it.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
CATALOG = "https://api.runpod.io/v2/catalog/gpus"
UA = "Mozilla/5.0 strata-dsv41-devgpu"
CUDA_VERSIONS = "12.8,12.9,13.0"
# the values REST v1 accepts in allowedCudaVersions (its PodCreateInput schema, 2026-10-14)
REST_CUDA_VERSIONS = {"13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"}


def key():
    k = os.environ.get("RUNPOD_API_KEY", "")
    if not k:
        print("runpod_api.py: RUNPOD_API_KEY is not set: export your RunPod API key",
              file=sys.stderr)
        sys.exit(1)
    return k


def request(method, path, body=None, base=REST):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    for k, v in {"Authorization": "Bearer " + key(), "User-Agent": UA, "Content-Type": "application/json",
                 "Accept": "application/json"}.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            text = r.read().decode()
            return r.status, (json.loads(text) if text.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]


def public_key():
    path = os.path.expanduser("~/.ssh/runpod.pub")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        print(f"runpod_api.py: {path}: {e} (the pod's SSH key pair is ~/.ssh/runpod)", file=sys.stderr)
        sys.exit(1)


def body(a, pub):
    b = {"name": "strata-dsv41-devgpu", "cloudType": a.cloud, "computeType": "GPU", "gpuTypeIds": [a.gpu],
         "gpuCount": 1, "imageName": a.image, "containerDiskInGb": a.disk, "volumeInGb": 0,
         "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub}}
    # REST v1 refuses (HTTP 400) any allowedCudaVersions entry above 13.0, while the catalog already lists 13.2 hosts:
    # with a newer minor in the list the field is left out (any host), and the pod's CUDA probe rejects a host whose
    # driver cannot run the payload
    versions = a.cuda_versions.split(",")
    if all(v in REST_CUDA_VERSIONS for v in versions):
        b["allowedCudaVersions"] = versions
    if a.dc:
        b["dataCenterIds"] = [a.dc]
    if a.min_ram > 0:
        b["minRAMPerGPU"] = a.min_ram
    if a.network_volume:
        b["networkVolumeId"] = a.network_volume
        b["volumeMountPath"] = "/workspace"
    return b


def price(a):
    query = "?include=AVAILABILITY&product=POD&cloud=%s&count=1&cudaVersions=%s" % (a.cloud, a.cuda_versions)
    status, d = request("GET", query, base=CATALOG)
    if status != 200 or not isinstance(d, dict) or not isinstance(d.get("gpus"), list):
        print(f"runpod_api.py: price {a.gpu} ({a.cloud}): HTTP {status} {str(d)[:200]}", file=sys.stderr)
        return 1
    key = a.cloud.lower()
    for g in d["gpus"]:
        if g.get("id") != a.gpu:
            continue
        # offered in this cloud: its secure / community flag (a price can be listed for a cloud that does not offer it)
        list_price = (g.get("price") or {}).get(key)
        offered = g.get(key, bool(list_price))
        avail = str(g.get("availability") or "NONE").upper()
        print(json.dumps({"gpu": a.gpu, "cloud": a.cloud, "price": list_price, "availability": avail}))
        return 0 if offered and list_price and avail != "NONE" else 3
    print(json.dumps({"gpu": a.gpu, "cloud": a.cloud, "price": None, "availability": "NONE"}))
    return 3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("price")
    s.add_argument("--gpu", required=True)
    s.add_argument("--cloud", required=True, choices=["SECURE", "COMMUNITY"])
    s.add_argument("--cuda-versions", default=CUDA_VERSIONS)
    for name in ("body", "create"):
        s = sub.add_parser(name)
        s.add_argument("--gpu", required=True)
        s.add_argument("--cloud", required=True, choices=["SECURE", "COMMUNITY"])
        s.add_argument("--dc", default="")
        s.add_argument("--image", required=True)
        s.add_argument("--disk", type=int, required=True)
        s.add_argument("--min-ram", type=int, default=0)
        s.add_argument("--network-volume", default="")
        s.add_argument("--cuda-versions", default=CUDA_VERSIONS)
    for name in ("get", "delete"):
        sub.add_parser(name).add_argument("id")
    a = p.parse_args()
    if getattr(a, "network_volume", "") and not a.dc:
        p.error("--network-volume needs --dc: a network volume lives in one data center")

    if a.cmd == "price":
        return price(a)
    if a.cmd == "body":
        print(json.dumps(body(a, "<~/.ssh/runpod.pub>")))
        return 0
    if a.cmd == "create":
        status, d = request("POST", "/pods", body(a, public_key()))
        if status in (200, 201) and isinstance(d, dict) and d.get("id"):
            print(json.dumps({"id": d["id"], "costPerHr": d.get("costPerHr"), "gpu": a.gpu, "cloud": a.cloud,
                              "dc": a.dc or "any", "machineId": d.get("machineId"), "memoryInGb": d.get("memoryInGb"),
                              "vcpuCount": d.get("vcpuCount")}))
            return 0
        text = d if isinstance(d, str) else json.dumps(d)
        print(f"runpod_api.py: create {a.gpu} ({a.cloud}, {a.dc or 'any DC'}): HTTP {status} {text[:200]}",
              file=sys.stderr)
        return 3 if ("no instances" in text or "could not find" in text) else 1
    if a.cmd == "get":
        status, d = request("GET", f"/pods/{a.id}")
        if status != 200 or not isinstance(d, dict):
            print(f"runpod_api.py: get {a.id}: HTTP {status} {d}", file=sys.stderr)
            return 1
        pm = d.get("portMappings") or {}
        print(json.dumps({"id": a.id, "desiredStatus": d.get("desiredStatus"), "publicIp": d.get("publicIp") or "",
                          "port22": pm.get("22") or "", "costPerHr": d.get("costPerHr"),
                          "memoryInGb": d.get("memoryInGb")}))
        return 0
    status, d = request("DELETE", f"/pods/{a.id}")
    if status in (200, 204, 404):
        print(f"deleted {a.id} (HTTP {status})")
        return 0
    print(f"runpod_api.py: delete {a.id}: HTTP {status} {d}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
