"""Build and curate a clean multi-font prototype database for EmbeddingOCR.

Embeds 62 alphanumeric classes across 25+ standard typographic clean fonts,
and saves the FAISS index + metadata to disk for instant loading.

Usage:
    uv run --no-sync python scripts/build_prototype_cache.py --num-fonts 25 --dim 768
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from embedding_ocr.data.glyph_renderer import ALPHANUMERIC_CLASSES
from embedding_ocr.data.ood_dataset import discover_fonts
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.index import FaissIndex
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and persist clean multi-font prototype index."
    )
    parser.add_argument(
        "--font-catalog",
        type=Path,
        default=Path("data/standard_clean_fonts.txt"),
    )
    parser.add_argument(
        "--num-fonts", type=int, default=25, help="Number of clean fonts to include."
    )
    parser.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/prototypes"),
    )
    parser.add_argument(
        "--index-backend",
        default="faiss_ivf_sq8",
        choices=["faiss_ivf_sq8", "faiss_sq8", "faiss", "faiss_flat_ip"],
        help="FAISS storage mode; 8-bit modes shrink the saved index.",
    )
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--quantizer-bits", type=int, default=8)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Max vectors used to train the quantizer (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("BUILDING CURATED MULTI-FONT PROTOTYPE DATABASE")
    print("=" * 80)

    if not args.font_catalog.exists():
        msg = f"Font catalog {args.font_catalog} not found."
        raise FileNotFoundError(msg)

    valid_fonts = [
        line.strip()
        for line in args.font_catalog.read_text().splitlines()
        if line.strip() and not line.startswith("#") and Path(line.strip()).exists()
    ]

    if not valid_fonts:
        print(
            f"No fonts from {args.font_catalog} are installed; "
            "falling back to discovered Latin fonts."
        )
        valid_fonts = discover_fonts("latin", limit=max(args.num_fonts, 1))

    if not valid_fonts:
        msg = (
            "No Latin fonts found. Install fonts-dejavu-core/fonts-liberation "
            "or pass a font catalog with existing paths."
        )
        raise FileNotFoundError(msg)

    selected_fonts = valid_fonts[: args.num_fonts]
    if len(selected_fonts) < args.num_fonts:
        print(
            f"Only {len(selected_fonts)} fonts available (requested {args.num_fonts})."
        )
    total_classes = len(ALPHANUMERIC_CLASSES)
    total_prototypes = len(selected_fonts) * total_classes

    print(f"Fonts selected: {len(selected_fonts)}")
    print(f"Classes:        {total_classes} (Alphanumeric: a-z, A-Z, 0-9)")
    print(f"Total vectors:  {total_prototypes}")
    print(f"Embedding Dim:  {args.dim}d")
    print(f"Index backend:  {args.index_backend}")

    # Initialize embedder and index
    embedder = EmbeddingGemma2(dim=args.dim, device=args.device, batch_size=0)
    index = FaissIndex(
        dim=args.dim,
        embedder_name=embedder.name,
        backend=args.index_backend,
        nlist=args.nlist,
        nprobe=args.nprobe,
        quantizer_bits=args.quantizer_bits,
        train_size=args.train_size,
    )
    proto_db = PrototypeDB(embedder=embedder, index=index, classes=ALPHANUMERIC_CLASSES)

    t0 = time.perf_counter()
    proto_db.build_from_fonts(selected_fonts)
    build_time = time.perf_counter() - t0

    save_path = args.output_dir / f"clean_alnum_{args.dim}d"
    proto_db.save(save_path)

    print(
        f"\nBuilt {total_prototypes} prototypes in {build_time:.2f}s ({build_time / total_prototypes * 1000.0:.1f}ms/item)."
    )
    print(
        f"Persisted to: {save_path.with_suffix('.faiss')} and {save_path.with_suffix('.labels.json')}"
    )
    print("Ready for zero-latency instant loading across all OCR benchmarks!")


if __name__ == "__main__":
    main()
