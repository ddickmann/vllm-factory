#!/usr/bin/env python3
"""End-to-end parity validation via real `vllm serve` + HTTP requests.

For each of the 12 plugins:
1. Start `vllm serve <model> --io-processor-plugin <name> ...`
2. Wait for /health to return 200
3. POST /pooling with {"data": {...}} through the IOProcessor pipeline
4. Compare response against saved reference results
5. Kill server, report PASS/FAIL
"""
import argparse
import base64
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "reference_results"
PORT = 9999
BASE_URL = f"http://localhost:{PORT}"

RONALDO_TEXT = (
    "Cristiano Ronaldo dos Santos Aveiro born 5 February 1985 is a Portuguese "
    "professional footballer who plays as a forward for and captains both "
    "Saudi Pro League club Al Nassr and the Portugal national team. Widely "
    "regarded as one of the greatest players of all time, Ronaldo has won "
    "five Ballon d Or awards, a record three UEFA Men Player of the Year "
    "Awards, and four European Golden Shoes, the most by a European player. "
    "He has scored the most goals in the history of the Champions League, "
    "is the all-time top scorer in UEFA Champions Leagues, the UEFA European "
    "Championship, and the UEFA Nations League."
)

PII_TEXT = (
    "Please contact John Smith at john.smith@example.com or call 555-123-4567. "
    "He lives at 123 Main Street, San Francisco, CA 94105. His social security "
    "number is 123-45-6789 and his credit card is 4532-1234-5678-9012. "
    "John works at NVIDIA Corporation in the AI research department."
)

GLINER2_TEXT = (
    "John Smith works at NVIDIA Corporation in Santa Clara, California. "
    "His email is john.smith@nvidia.com and phone number is 555-123-4567. "
    "He is the VP of AI Research and reports to Jensen Huang."
)

LINKER_TEXT = (
    "Apple announced new products in California. "
    "Michael Jordan joined the team."
)

RERANKER_TEXT = (
    "Farnese Palace is one of the most important palaces in the city of Rome. "
    "It was designed by Antonio da Sangallo the Younger in 1517 for the "
    "Farnese family. Michelangelo also contributed to its design. Today it "
    "serves as the French Embassy in Italy."
)


# ── Comparison helpers ──────────────────────────────────────────────────

def cosine_sim(a, b):
    a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def entity_prf(ref_entities, pred_entities):
    """Precision, recall, F1 over (text, label) tuples."""
    ref_set = {(e["text"].lower().strip(), e["label"].lower().strip()) for e in ref_entities}
    pred_set = {(e["text"].lower().strip(), e["label"].lower().strip()) for e in pred_entities}
    if not ref_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not ref_set or not pred_set:
        return 0.0, 0.0, 0.0
    tp = len(ref_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0
    rec = tp / len(ref_set) if ref_set else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def entity_f1(ref_entities, pred_entities):
    """F1 over (text, label) tuples."""
    _, _, f1 = entity_prf(ref_entities, pred_entities)
    return f1


def entity_score_parity(ref_entities, pred_entities):
    """Compare confidence scores for matched (text, label) pairs.

    Returns (mean_delta, max_delta, n_compared). Informational only.
    """
    ref_by_key = {}
    for e in ref_entities:
        key = (e["text"].lower().strip(), e["label"].lower().strip())
        if "score" in e:
            ref_by_key[key] = e["score"]
    deltas = []
    for e in pred_entities:
        key = (e["text"].lower().strip(), e["label"].lower().strip())
        if key in ref_by_key and "score" in e:
            deltas.append(abs(ref_by_key[key] - e["score"]))
    if not deltas:
        return 0.0, 0.0, 0
    return sum(deltas) / len(deltas), max(deltas), len(deltas)


def linked_entity_match(ref_entities, pred_entities, score_atol=0.1):
    """Check linked entities match by text/label, with score tolerance."""
    ref_by_text = {e["text"].lower().strip(): e for e in ref_entities}
    matched = 0
    score_deltas = []
    for pe in pred_entities:
        key = pe["text"].lower().strip()
        if key in ref_by_text:
            re = ref_by_text[key]
            if pe.get("label", "").lower() == re.get("label", "").lower():
                matched += 1
                if "score" in pe and "score" in re:
                    score_deltas.append(abs(pe["score"] - re["score"]))
    total = max(len(ref_entities), len(pred_entities), 1)
    match_ratio = matched / total
    scores_ok = all(d <= score_atol for d in score_deltas) if score_deltas else True
    return match_ratio, scores_ok, score_deltas


# ── Server lifecycle ────────────────────────────────────────────────────

def start_server(model: str, io_plugin: str, extra_args: list[str],
                 env_extra: dict | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--io-processor-plugin", io_plugin,
        "--port", str(PORT),
        "--trust-remote-code",
        "--enforce-eager",
    ] + extra_args

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, start_new_session=True,
    )
    return proc


def wait_healthy(timeout: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(3)
    return False


def kill_server(proc: subprocess.Popen):
    if proc.poll() is None:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    time.sleep(2)


def send_pooling(data: dict, model_name: str = "model") -> dict:
    body = {"model": model_name, "data": data}
    r = requests.post(f"{BASE_URL}/pooling", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


# ── Plugin test configs ─────────────────────────────────────────────────

def _ner_serve_flags():
    return ["--dtype", "bfloat16", "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill"]


def _colbert_serve_flags():
    return ["--dtype", "bfloat16", "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill"]


def _multimodal_serve_flags():
    return ["--dtype", "bfloat16", "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill"]


PLUGINS: list[dict[str, Any]] = [
    # 1. embeddinggemma (300M) - smallest, test first
    {
        "name": "embeddinggemma",
        "model": "unsloth/embeddinggemma-300m",
        "io_plugin": "embeddinggemma_io",
        "serve_flags": ["--dtype", "bfloat16", "--no-enable-prefix-caching",
                        "--no-enable-chunked-prefill"],
        "timeout": 120,
        "test_fn": "test_embeddinggemma",
    },
    # 2. mmbert_gliner (150M)
    {
        "name": "mmbert_gliner",
        "model": "/tmp/sauerkraut-gliner-vllm",
        "io_plugin": "mmbert_gliner_io",
        "serve_flags": _ner_serve_flags(),
        "timeout": 120,
        "test_fn": "test_ner_gliner",
        "test_kwargs": {
            "ref_file": "mmbert_gliner/mmbert-gliner-reference.json",
            "text": RONALDO_TEXT,
            "labels": ["person", "award", "date", "competitions", "teams"],
        },
    },
    # 3. deberta_gliner (300M)
    {
        "name": "deberta_gliner",
        "model": "/tmp/gliner-pii-vllm",
        "io_plugin": "deberta_gliner_io",
        "serve_flags": _ner_serve_flags(),
        "timeout": 120,
        "test_fn": "test_ner_gliner",
        "test_kwargs": {
            "ref_file": "deberta_gliner/gliner-pii-reference.json",
            "text": PII_TEXT,
            "labels": ["person", "email", "phone_number", "address", "organization"],
        },
    },
    # 4. deberta_gliner2 (350M)
    {
        "name": "deberta_gliner2",
        "model": "/tmp/gliner2-vllm",
        "io_plugin": "deberta_gliner2_io",
        "serve_flags": _ner_serve_flags(),
        "timeout": 120,
        "test_fn": "test_deberta_gliner2",
    },
    # 5. mt5_gliner (1.2B) — T5 requires bfloat16 to avoid fp16 NaN in relative position bias
    {
        "name": "mt5_gliner",
        "model": "/tmp/gliner-x-large-vllm",
        "io_plugin": "mt5_gliner_io",
        "serve_flags": ["--dtype", "bfloat16", "--no-enable-prefix-caching",
                        "--no-enable-chunked-prefill"],
        "timeout": 180,
        "test_fn": "test_ner_gliner",
        "test_kwargs": {
            "ref_file": "mt5_gliner/mt5-gliner-reference.json",
            "text": RONALDO_TEXT,
            "labels": ["person", "award", "date", "competitions", "teams"],
        },
    },
    # 6. deberta_gliner_linker (400M)
    {
        "name": "deberta_gliner_linker",
        "model": str(REPO_ROOT / "plugins" / "deberta_gliner_linker" / "_model_cache"),
        "io_plugin": "deberta_gliner_linker_io",
        "serve_flags": _ner_serve_flags(),
        "timeout": 180,
        "test_fn": "test_gliner_linker",
    },
    # 7. modernbert_gliner_rerank (150M)
    {
        "name": "modernbert_gliner_rerank",
        "model": str(REPO_ROOT / "plugins" / "modernbert_gliner_rerank" / "_model_cache"),
        "io_plugin": "modernbert_gliner_rerank_io",
        "serve_flags": _ner_serve_flags(),
        "timeout": 180,
        "test_fn": "test_gliner_rerank",
    },
    # 8. moderncolbert (150M)
    {
        "name": "moderncolbert",
        "model": "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
        "io_plugin": "moderncolbert_io",
        "serve_flags": _colbert_serve_flags(),
        "timeout": 120,
        "test_fn": "test_moderncolbert",
    },
    # 9. lfm2_colbert (350M)
    {
        "name": "lfm2_colbert",
        "model": "LiquidAI/LFM2-ColBERT-350M",
        "io_plugin": "lfm2_colbert_io",
        "serve_flags": _colbert_serve_flags(),
        "timeout": 120,
        "test_fn": "test_lfm2_colbert",
    },
    # 10. collfm2 (450M)
    {
        "name": "collfm2",
        "model": "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
        "io_plugin": "collfm2_io",
        "serve_flags": _multimodal_serve_flags(),
        "timeout": 180,
        "test_fn": "test_collfm2",
    },
    # 11. colqwen3 (1.7B) — limit max-model-len to avoid OOM on 24GB GPU
    {
        "name": "colqwen3",
        "model": "VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
        "io_plugin": "colqwen3_io",
        "serve_flags": _multimodal_serve_flags() + ["--max-model-len", "8192",
                        "--limit-mm-per-prompt", '{"image": 1}'],
        "timeout": 240,
        "test_fn": "test_colqwen3",
    },
    # 12. nemotron_colembed (4B) - largest, last
    {
        "name": "nemotron_colembed",
        "model": "nvidia/nemotron-colembed-vl-4b-v2",
        "io_plugin": "nemotron_colembed_io",
        "serve_flags": _multimodal_serve_flags(),
        "timeout": 360,
        "test_fn": "test_nemotron_colembed",
    },
]


# ── Per-plugin test functions ───────────────────────────────────────────

def test_embeddinggemma(plugin_cfg: dict) -> tuple[bool, str]:
    ref = torch.load(REF_DIR / "embeddinggemma" / "embeddinggemma-reference.pt",
                     weights_only=False)
    ref_texts = ref["texts"]
    ref_emb = ref["embeddings"]  # (N, 768)

    task_map = {
        "task: search result | query:": "query",
        "task: sentence similarity | query:": "STS",
        "task: clustering | query:": "Clustering",
        "title: none | text:": "document",
    }

    results = []
    for i, text in enumerate(ref_texts):
        task = "query"
        raw_text = text
        for prefix, t in task_map.items():
            if text.startswith(prefix):
                task = t
                raw_text = text[len(prefix):].strip()
                break

        resp = send_pooling({"text": raw_text, "task": task}, plugin_cfg["model"])
        pred_emb = resp["data"]
        results.append(pred_emb)

    sims = []
    for i, pred in enumerate(results):
        sim = cosine_sim(pred, ref_emb[i].tolist())
        sims.append(sim)

    avg_sim = sum(sims) / len(sims)
    passed = avg_sim >= 0.99
    return passed, f"avg_cosine={avg_sim:.4f} (threshold=0.99)"


def test_ner_gliner(plugin_cfg: dict, ref_file: str, text: str,
                    labels: list[str]) -> tuple[bool, str]:
    ref_path = REF_DIR / ref_file
    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_entities = ref_data["entities"]

    resp = send_pooling({"text": text, "labels": labels, "threshold": 0.3,
                         "flat_ner": True},
                        plugin_cfg["model"])
    pred_entities = resp["data"]

    prec, rec, f1 = entity_prf(ref_entities, pred_entities)
    mean_d, max_d, n_cmp = entity_score_parity(ref_entities, pred_entities)
    passed = rec >= 0.95
    detail = (f"recall={rec:.3f} prec={prec:.3f} F1={f1:.3f}, "
              f"ref={len(ref_entities)} pred={len(pred_entities)}, "
              f"score_delta={mean_d:.3f} (max={max_d:.3f}, n={n_cmp})")
    return passed, detail


def test_deberta_gliner2(plugin_cfg: dict) -> tuple[bool, str]:
    ref_path = REF_DIR / "deberta_gliner2" / "gliner2-reference.json"
    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_entities_dict = ref_data["entities"]["entities"]

    labels = ["person", "organization", "location", "email", "phone_number"]
    resp = send_pooling({"text": GLINER2_TEXT, "labels": labels}, plugin_cfg["model"])
    pred = resp["data"]

    ref_flat = []
    for label, ents in ref_entities_dict.items():
        for e in ents:
            ref_flat.append({"text": e["text"], "label": label,
                             **({"score": e["confidence"]} if "confidence" in e else {})})

    pred_flat = []
    if isinstance(pred, dict) and "entities" in pred:
        ent_dict = pred["entities"]
        if isinstance(ent_dict, dict):
            for label, ents in ent_dict.items():
                if isinstance(ents, list):
                    for e in ents:
                        if isinstance(e, dict):
                            pred_flat.append({"text": e.get("text", ""), "label": label,
                                              **({"score": e["confidence"]}
                                                 if "confidence" in e else {})})
                        elif isinstance(e, str):
                            pred_flat.append({"text": e, "label": label})
    elif isinstance(pred, list):
        pred_flat = pred

    prec, rec, f1 = entity_prf(ref_flat, pred_flat)
    mean_d, max_d, n_cmp = entity_score_parity(ref_flat, pred_flat)
    passed = rec >= 0.95
    detail = (f"recall={rec:.3f} prec={prec:.3f} F1={f1:.3f}, "
              f"ref={len(ref_flat)} pred={len(pred_flat)}, "
              f"score_delta={mean_d:.3f} (max={max_d:.3f}, n={n_cmp})")
    return passed, detail


def test_gliner_linker(plugin_cfg: dict) -> tuple[bool, str]:
    ref_path = REF_DIR / "deberta_gliner_linker" / "glinker-linker-reference.json"
    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_entities = ref_data["entities"]

    labels = ["person", "company", "location"]
    resp = send_pooling({"text": LINKER_TEXT, "labels": labels, "threshold": 0.3},
                        plugin_cfg["model"])
    pred_entities = resp["data"]

    prec, rec, f1 = entity_prf(ref_entities, pred_entities)
    match_ratio, scores_ok, score_deltas = linked_entity_match(ref_entities, pred_entities)
    mean_d, max_d, n_cmp = entity_score_parity(ref_entities, pred_entities)
    passed = rec >= 0.95 and match_ratio >= 0.5
    detail = (f"recall={rec:.3f} prec={prec:.3f} F1={f1:.3f}, "
              f"link_match={match_ratio:.3f}, "
              f"score_delta={mean_d:.3f} (max={max_d:.3f}, n={n_cmp})")
    return passed, detail


def test_gliner_rerank(plugin_cfg: dict) -> tuple[bool, str]:
    ref_path = REF_DIR / "modernbert_gliner_rerank" / "glinker-rerank-reference.json"
    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_entities = ref_data["entities"]

    labels = ref_data.get("labels", ["person", "building", "city", "country", "family"])
    threshold = ref_data.get("threshold", 0.3)
    resp = send_pooling({"text": RERANKER_TEXT, "labels": labels, "threshold": threshold,
                         "flat_ner": True}, plugin_cfg["model"])
    pred_entities = resp["data"]

    prec, rec, f1 = entity_prf(ref_entities, pred_entities)
    mean_d, max_d, n_cmp = entity_score_parity(ref_entities, pred_entities)
    passed = rec >= 0.85
    detail = (f"recall={rec:.3f} prec={prec:.3f} F1={f1:.3f}, "
              f"ref={len(ref_entities)} pred={len(pred_entities)}, "
              f"score_delta={mean_d:.3f} (max={max_d:.3f}, n={n_cmp})")
    return passed, detail


def test_moderncolbert(plugin_cfg: dict) -> tuple[bool, str]:
    ref_dir = REF_DIR / "moderncolbert"
    with open(ref_dir / "queries.json") as f:
        queries = json.load(f)
    with open(ref_dir / "documents.json") as f:
        documents = json.load(f)
    ref_q_emb = torch.load(ref_dir / "query_embeddings.pt", weights_only=False)
    ref_d_emb = torch.load(ref_dir / "document_embeddings.pt", weights_only=False)

    sims = []
    for i, q in enumerate(queries):
        resp = send_pooling({"text": q, "is_query": True}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        ref = ref_q_emb[i].float()
        n = min(pred.shape[0] // ref.shape[1], ref.shape[0]) if pred.dim() == 1 else min(pred.shape[0], ref.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * ref.shape[1]].reshape(n, ref.shape[1])
        else:
            pred = pred[:n]
        ref = ref[:n]
        sim = torch.nn.functional.cosine_similarity(pred, ref, dim=-1).mean().item()
        sims.append(sim)

    for i, d in enumerate(documents):
        resp = send_pooling({"text": d, "is_query": False}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        ref = ref_d_emb[i].float()
        n = min(pred.shape[0] // ref.shape[1], ref.shape[0]) if pred.dim() == 1 else min(pred.shape[0], ref.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * ref.shape[1]].reshape(n, ref.shape[1])
        else:
            pred = pred[:n]
        ref = ref[:n]
        sim = torch.nn.functional.cosine_similarity(pred, ref, dim=-1).mean().item()
        sims.append(sim)

    avg = sum(sims) / len(sims)
    passed = avg >= 0.95
    return passed, f"avg_cosine={avg:.4f} (threshold=0.95), {len(sims)} embeddings"


def test_lfm2_colbert(plugin_cfg: dict) -> tuple[bool, str]:
    ref = torch.load(REF_DIR / "lfm2_colbert" / "lfm2-colbert-reference.pt",
                     weights_only=False)
    queries = ref["queries"]
    documents = ref["documents"]
    ref_q = ref["query_embeddings"]
    ref_d = ref["document_embeddings"]

    sims = []
    for i, q in enumerate(queries):
        resp = send_pooling({"text": q}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_q[i].float()
        n = min(pred.shape[0] // r.shape[1], r.shape[0]) if pred.dim() == 1 else min(pred.shape[0], r.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * r.shape[1]].reshape(n, r.shape[1])
        else:
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    for i, d in enumerate(documents):
        resp = send_pooling({"text": d}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_d[i].float()
        n = min(pred.shape[0] // r.shape[1], r.shape[0]) if pred.dim() == 1 else min(pred.shape[0], r.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * r.shape[1]].reshape(n, r.shape[1])
        else:
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    avg = sum(sims) / len(sims)
    passed = avg >= 0.95
    return passed, f"avg_cosine={avg:.4f} (threshold=0.95), {len(sims)} embeddings"


def test_collfm2(plugin_cfg: dict) -> tuple[bool, str]:
    ref_dir = REF_DIR / "collfm2"
    meta = json.loads((ref_dir / "reference_metadata.json").read_text())
    ref_q = torch.load(ref_dir / "reference_query_embeddings.pt", weights_only=False)

    queries = meta["queries"]
    sims = []
    for i, q in enumerate(queries):
        resp = send_pooling({"text": q, "is_query": True}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_q[i].float()
        n = min(pred.shape[0] // r.shape[1], r.shape[0]) if pred.dim() == 1 else min(pred.shape[0], r.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * r.shape[1]].reshape(n, r.shape[1])
        else:
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    ref_ie = torch.load(ref_dir / "reference_image_embeddings.pt", weights_only=False)
    for i in range(meta["n_images"]):
        img_path = ref_dir / f"test_image_{i}.png"
        img_b64 = base64.b64encode(img_path.read_bytes()).decode()
        data_uri = f"data:image/png;base64,{img_b64}"
        resp = send_pooling({"image": data_uri, "is_query": False}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_ie[i].float()
        n = min(pred.shape[0] // r.shape[1], r.shape[0]) if pred.dim() == 1 else min(pred.shape[0], r.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * r.shape[1]].reshape(n, r.shape[1])
        else:
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    avg = sum(sims) / len(sims)
    passed = avg >= 0.90
    return passed, f"avg_cosine={avg:.4f} (threshold=0.90), {len(sims)} embeddings"


def test_colqwen3(plugin_cfg: dict) -> tuple[bool, str]:
    ref_dir = REF_DIR / "colqwen3"
    with open(ref_dir / "queries.json") as f:
        queries = json.load(f)
    ref_q = torch.load(ref_dir / "query_embeddings.pt", weights_only=False)

    sims = []
    for i, q in enumerate(queries):
        resp = send_pooling({"text": q, "is_query": True}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_q[i].float()
        n = min(pred.shape[0] // r.shape[1], r.shape[0]) if pred.dim() == 1 else min(pred.shape[0], r.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * r.shape[1]].reshape(n, r.shape[1])
        else:
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    ref_ie = torch.load(ref_dir / "image_embeddings.pt", weights_only=False)
    images = sorted((ref_dir / "images").iterdir())
    for i, img_path in enumerate(images):
        img_b64 = base64.b64encode(img_path.read_bytes()).decode()
        data_uri = f"data:image/png;base64,{img_b64}"
        resp = send_pooling({"image": data_uri, "is_query": False}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_ie[i].float()
        n = min(pred.shape[0] // r.shape[1], r.shape[0]) if pred.dim() == 1 else min(pred.shape[0], r.shape[0])
        if pred.dim() == 1:
            pred = pred[:n * r.shape[1]].reshape(n, r.shape[1])
        else:
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    avg = sum(sims) / len(sims)
    passed = avg >= 0.90
    return passed, f"avg_cosine={avg:.4f} (threshold=0.90), {len(sims)} embeddings"


def test_nemotron_colembed(plugin_cfg: dict) -> tuple[bool, str]:
    ref = torch.load(REF_DIR / "nemotron_colembed" / "nemotron-colembed-reference.pt",
                     weights_only=False)
    queries = ref["queries"]
    image_urls = ref["image_urls"]
    ref_q = ref["query_embeddings"]  # (3, 32, 2560)
    ref_ie = ref["image_embeddings"]  # (3, 791, 2560)

    sims = []
    for i, q in enumerate(queries):
        resp = send_pooling({"text": q, "is_query": True}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_q[i].float()
        if pred.dim() == 1:
            dim = r.shape[-1]
            n = min(pred.shape[0] // dim, r.shape[0])
            pred = pred[:n * dim].reshape(n, dim)
        else:
            n = min(pred.shape[0], r.shape[0])
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    for i, url in enumerate(image_urls):
        resp = send_pooling({"image": url, "is_query": False}, plugin_cfg["model"])
        pred = torch.tensor(resp["data"], dtype=torch.float32)
        r = ref_ie[i].float()
        if pred.dim() == 1:
            dim = r.shape[-1]
            n = min(pred.shape[0] // dim, r.shape[0])
            pred = pred[:n * dim].reshape(n, dim)
        else:
            n = min(pred.shape[0], r.shape[0])
            pred = pred[:n]
        r = r[:n]
        sim = torch.nn.functional.cosine_similarity(pred, r, dim=-1).mean().item()
        sims.append(sim)

    avg = sum(sims) / len(sims)
    passed = avg >= 0.90
    return passed, f"avg_cosine={avg:.4f} (threshold=0.90), {len(sims)} embeddings"


# ── Test dispatcher ─────────────────────────────────────────────────────

TEST_FUNCTIONS = {
    "test_embeddinggemma": test_embeddinggemma,
    "test_ner_gliner": test_ner_gliner,
    "test_deberta_gliner2": test_deberta_gliner2,
    "test_gliner_linker": test_gliner_linker,
    "test_gliner_rerank": test_gliner_rerank,
    "test_moderncolbert": test_moderncolbert,
    "test_lfm2_colbert": test_lfm2_colbert,
    "test_collfm2": test_collfm2,
    "test_colqwen3": test_colqwen3,
    "test_nemotron_colembed": test_nemotron_colembed,
}


def run_plugin(plugin_cfg: dict) -> tuple[bool, str]:
    name = plugin_cfg["name"]
    model = plugin_cfg["model"]
    io_plugin = plugin_cfg["io_plugin"]
    serve_flags = plugin_cfg["serve_flags"]
    timeout = plugin_cfg.get("timeout", 180)
    test_fn_name = plugin_cfg["test_fn"]
    test_kwargs = plugin_cfg.get("test_kwargs", {})

    print(f"\n{'='*70}")
    print(f"  TESTING: {name}")
    print(f"  Model: {model}")
    print(f"  IO Plugin: {io_plugin}")
    print(f"{'='*70}")

    proc = start_server(model, io_plugin, serve_flags)
    try:
        print(f"  Waiting for server (timeout={timeout}s)...")
        if not wait_healthy(timeout):
            out = ""
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
            return False, f"Server failed to start within {timeout}s. Output tail:\n{out[-2000:] if out else '(process still running)'}"

        print("  Server healthy. Running test...")
        fn = TEST_FUNCTIONS[test_fn_name]
        if test_kwargs:
            passed, detail = fn(plugin_cfg, **test_kwargs)
        else:
            passed, detail = fn(plugin_cfg)
        return passed, detail
    except Exception as e:
        return False, f"Exception: {e}"
    finally:
        print(f"  Killing server for {name}...")
        kill_server(proc)
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="End-to-end parity validation via vllm serve")
    parser.add_argument("--plugin", type=str, help="Run only this plugin (by name)")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Skip these plugins")
    parser.add_argument("--start-from", type=str, help="Start from this plugin")
    args = parser.parse_args()

    plugins_to_test = PLUGINS
    if args.plugin:
        plugins_to_test = [p for p in PLUGINS if p["name"] == args.plugin]
        if not plugins_to_test:
            print(f"Unknown plugin: {args.plugin}")
            sys.exit(1)
    elif args.start_from:
        idx = next((i for i, p in enumerate(PLUGINS) if p["name"] == args.start_from), None)
        if idx is None:
            print(f"Unknown plugin: {args.start_from}")
            sys.exit(1)
        plugins_to_test = PLUGINS[idx:]

    if args.skip:
        plugins_to_test = [p for p in plugins_to_test if p["name"] not in args.skip]

    results: list[tuple[str, bool, str]] = []

    print(f"Testing {len(plugins_to_test)} plugins via real vllm serve + HTTP requests\n")
    for plugin_cfg in plugins_to_test:
        passed, detail = run_plugin(plugin_cfg)
        status = "PASS" if passed else "FAIL"
        results.append((plugin_cfg["name"], passed, detail))
        print(f"\n  >>> {plugin_cfg['name']}: {status} — {detail}")

    print(f"\n\n{'='*70}")
    print("  FINAL RESULTS")
    print(f"{'='*70}")
    all_pass = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name:30s} {detail}")

    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} plugins passed")

    if all_pass:
        print("\n  ALL PLUGINS PASSED")
    else:
        print("\n  SOME PLUGINS FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
