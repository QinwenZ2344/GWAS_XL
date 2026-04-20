#!/usr/bin/env python3

import argparse
import csv
import gzip
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frozen text embeddings from IID+Text table using PubMedBERT (or any HF encoder)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input table path (.txt/.tsv/.csv and optional .gz).",
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        required=True,
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Hugging Face encoder model id.",
    )
    parser.add_argument("--iid_col", type=str, default="IID", help="ID column name.")
    parser.add_argument("--text_col", type=str, default="Text", help="Text column name.")
    parser.add_argument(
        "--keep_cols",
        type=str,
        default="",
        help="Comma-separated extra columns to keep in output metadata (e.g. diabetes,sex,age).",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help="Input delimiter. Default is tab.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy for token features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cuda:0/cpu. If None, auto-select.",
    )
    parser.add_argument(
        "--skip_empty_text",
        action="store_true",
        help="Skip rows with empty text. Default behavior keeps them as '[EMPTY_TEXT]'.",
    )
    return parser.parse_args()


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def count_data_rows(path: str) -> int:
    with open_text(path) as f:
        # Subtract header line.
        return max(sum(1 for _ in f) - 1, 0)


def batched(iterable: Iterator[dict], batch_size: int) -> Iterator[list[dict]]:
    bucket = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]

    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def main() -> None:
    args = parse_args()

    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    n_rows = count_data_rows(args.input_file)
    if n_rows == 0:
        raise ValueError(f"No data rows found in {args.input_file}")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    output_ids: list[str] = []
    output_embeddings: list[torch.Tensor] = []
    output_meta: dict[str, list[str]] = {col: [] for col in keep_cols}

    with open_text(args.input_file) as f:
        reader = csv.DictReader(f, delimiter=args.sep)
        if args.iid_col not in (reader.fieldnames or []):
            raise ValueError(f"iid_col '{args.iid_col}' not in file header: {reader.fieldnames}")
        if args.text_col not in (reader.fieldnames or []):
            raise ValueError(f"text_col '{args.text_col}' not in file header: {reader.fieldnames}")
        for col in keep_cols:
            if col not in (reader.fieldnames or []):
                raise ValueError(f"keep_col '{col}' not in file header: {reader.fieldnames}")

        progress = tqdm(batched(reader, args.batch_size), total=(n_rows + args.batch_size - 1) // args.batch_size)
        for batch_rows in progress:
            batch_ids = []
            batch_texts = []
            batch_keep: dict[str, list[str]] = {col: [] for col in keep_cols}

            for row in batch_rows:
                iid = str(row.get(args.iid_col, "")).strip()
                text = str(row.get(args.text_col, "")).strip()
                if not iid:
                    continue
                if not text:
                    if args.skip_empty_text:
                        continue
                    text = "[EMPTY_TEXT]"

                batch_ids.append(iid)
                batch_texts.append(text)
                for col in keep_cols:
                    batch_keep[col].append(str(row.get(col, "")))

            if not batch_texts:
                continue

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
            )
            emb = pool_embeddings(
                last_hidden_state=out.last_hidden_state,
                attention_mask=enc.get("attention_mask", None),
                pooling=args.pooling,
            ).cpu()

            output_ids.extend(batch_ids)
            output_embeddings.append(emb)
            for col in keep_cols:
                output_meta[col].extend(batch_keep[col])

    if not output_embeddings:
        raise RuntimeError("No embeddings were produced. Check input columns and text content.")

    embeddings = torch.cat(output_embeddings, dim=0)
    save_obj = {
        "ids": output_ids,
        "embeddings": embeddings,
        "model_name": args.model_name,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "text_col": args.text_col,
        "iid_col": args.iid_col,
        "keep_cols": output_meta,
        "input_file": str(Path(args.input_file).resolve()),
    }

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_obj, output_path)

    print(f"Saved: {output_path}")
    print(f"Number of samples: {len(output_ids)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
