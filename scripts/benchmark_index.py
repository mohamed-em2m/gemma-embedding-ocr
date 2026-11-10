"""Benchmark FAISS flat vs 8-bit scalar-quantized prototype indexes.

Reports index size, build time, search latency, and recall@k versus the exact
float32 ``IndexFlatIP`` baseline. Works on a saved prototype matrix or on
synthetic clustered unit vectors, so it needs no embedder or model download.

Examples:
    uv run --no-sync python scripts/benchmark_index.py --n 100000 --dim 768
    uv run --no-sync python scripts/benchmark_index.py \
        --vectors artifacts/vocab_cache/words_768d.npy --nprobe 32
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from embedding_ocr.recognition.embedding.index import FaissIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--backends",
        default="flat_ip,sq8,ivf_sq8",
        help="Comma-separated subset of flat_ip,sq8,ivf_sq8.",
    )
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def _synthetic_data(
    n: int, dim: int, clusters: int, num_queries: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    labels = rng.integers(0, clusters, size=n + num_queries)
    noise = rng.standard_normal((n + num_queries, dim)).astype(np.float32) * 0.15
    vectors = _normalize(centers[labels] + noise)
    return vectors[:n], vectors[n:]


def _load_vectors(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.vectors is None:
        return _synthetic_data(
            args.n, args.dim, args.clusters, args.num_queries, args.seed
        )
    vectors = _normalize(np.load(args.vectors))
    if args.queries is not None:
        queries = _normalize(np.load(args.queries))
    else:
        count = min(args.num_queries, len(vectors))
        queries = vectors[:count].copy()
    return vectors, queries


def _recall_at_k(predicted: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    for pred_row, truth_row in zip(predicted, truth, strict=False):
        hits += len(set(pred_row.tolist()) & set(truth_row.tolist()))
    return hits / max(1, predicted.size)


def _index_size_mb(index: FaissIndex, path: Path) -> float:
    index.save(path)
    index_file = path.with_suffix(".faiss")
    return index_file.stat().st_size / (1024 * 1024)


def main() -> None:
    args = parse_args()
    vectors, queries = _load_vectors(args)
    labels = np.arange(len(vectors))
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]

    print(
        f"Vectors: {vectors.shape[0]:,} x {vectors.shape[1]}d, "
        f"queries: {queries.shape[0]:,}, k={args.k}"
    )

    flat = FaissIndex(
        dim=vectors.shape[1],
        backend="flat_ip",
        omp_threads=args.omp_threads,
    )
    t0 = time.perf_counter()
    flat.build(vectors, labels)
    flat_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    truth_sims, truth_ids = flat.search(queries, k=args.k)
    flat_search = (time.perf_counter() - t0) * 1000.0 / len(queries)

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for backend in backends:
            index = FaissIndex(
                dim=vectors.shape[1],
                backend=backend,
                omp_threads=args.omp_threads,
                nlist=args.nlist,
                nprobe=args.nprobe,
                train_size=args.train_size,
            )
            t0 = time.perf_counter()
            index.build(vectors, labels)
            build_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            sims, ids = index.search(queries, k=args.k)
            search_ms = (time.perf_counter() - t0) * 1000.0 / len(queries)

            size_mb = _index_size_mb(index, Path(tmp) / backend)
            recall = 1.0 if backend == "flat_ip" else _recall_at_k(ids, truth_ids)
            mean_top1 = float(np.mean(sims[:, 0])) if sims.size else 0.0
            row = {
                "backend": backend,
                "build_s": build_s,
                "search_ms_per_query": search_ms,
                "recall_at_k": recall,
                "mean_top1_similarity": mean_top1,
                "size_mb": size_mb,
                "nlist": int(getattr(index._index, "nlist", 0)),
                "nprobe": int(getattr(index._index, "nprobe", 0)),
            }
            rows.append(row)
            print(
                f"{backend:>10s}  build={build_s:7.2f}s  "
                f"search={search_ms:8.3f}ms  recall@{args.k}={recall:6.3f}  "
                f"size={size_mb:8.2f}MB"
            )

    if args.backends.strip() == "flat_ip":
        print(f"flat reference: build={flat_build:.2f}s search={flat_search:.3f}ms")
    if args.output is not None:
        payload = {
            "n": int(vectors.shape[0]),
            "dim": int(vectors.shape[1]),
            "queries": int(queries.shape[0]),
            "k": args.k,
            "flat": {
                "build_s": flat_build,
                "search_ms_per_query": flat_search,
                "mean_top1_similarity": float(np.mean(truth_sims[:, 0])),
            },
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
