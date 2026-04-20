#!/usr/bin/env python3
"""
Create PubMedBERT embeddings for UKB smoking phenotype text.

Default input:
  data/ukb_phenotype/smoke_text_df.tsv

Default output:
  data/results/smoke_text_pubmedbert_embeddings.pt

Saved .pt object format:
{
  "ids": List[str],                     # person IID order aligned with embedding rows
  "iid_to_index": Dict[str, int],       # quick lookup from IID to row index
  "embeddings": FloatTensor[N, D],      # N persons x D embedding dimension
  "model_name": str,
  "pooling": str,
  "max_length": int,
  "input_file": str
}
"""

import argparse
import csv
import gzip
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    project_root = Path("/proj/yunligrp/users/qinwen/R_works/LLM_embedding")
    default_input = project_root / "data" / "ukb_phenotype" / "smoke_text_df.tsv"
    default_output = project_root / "data" / "results" / "smoke_text_pubmedbert_embeddings.pt"

    parser = argparse.ArgumentParser(
        description="Generate embeddings for UKB smoking text using neuml/pubmedbert-base-embeddings."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(default_input),
        help="Input TSV with columns IID and Text.",
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        default=str(default_output),
        help="Output .pt path.",
    )
    parser.add_argument(
        "--output_r_tsv_gz",
        type=str,
        default=None,
        help="Optional R-friendly output (.tsv.gz). Defaults to <output_pt stem>_r.tsv.gz.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="neuml/pubmedbert-base-embeddings",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for tokenization/model forward pass.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy over token embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cuda:0/cpu. If unset, auto-selects cuda when available.",
    )
    parser.add_argument(
        "--skip_empty_text",
        action="store_true",
        help="Skip records with empty text instead of replacing with [EMPTY_TEXT].",
    )
    return parser.parse_args()


def count_data_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8", newline="") as f:
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


def pool_embeddings(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None, pooling: str
) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]

    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def save_r_friendly_tsv_gz(output_path: Path, ids: list[str], embeddings: torch.Tensor) -> None:
    """
    Save embeddings as a long tabular matrix for easy R loading:
    IID, emb_0001, emb_0002, ..., emb_D
    """
    emb = embeddings.float().numpy()
    dim = emb.shape[1]
    header = ["IID"] + [f"emb_{i:04d}" for i in range(1, dim + 1)]

    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for iid, row in zip(ids, emb):
            writer.writerow([iid] + [f"{x:.8g}" for x in row])


@torch.no_grad()
def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_pt)
    output_r_path = (
        Path(args.output_r_tsv_gz)
        if args.output_r_tsv_gz
        else output_path.with_name(f"{output_path.stem}_r.tsv.gz")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    n_rows = count_data_rows(str(input_path))
    if n_rows == 0:
        raise ValueError(f"No data rows found in {input_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input file: {input_path}")
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    output_ids: list[str] = []
    output_embeddings: list[torch.Tensor] = []

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "IID" not in (reader.fieldnames or []):
            raise ValueError(f"Column 'IID' not in header: {reader.fieldnames}")
        if "Text" not in (reader.fieldnames or []):
            raise ValueError(f"Column 'Text' not in header: {reader.fieldnames}")

        total_batches = (n_rows + args.batch_size - 1) // args.batch_size
        progress = tqdm(batched(reader, args.batch_size), total=total_batches, desc="Embedding")

        for batch_rows in progress:
            batch_ids: list[str] = []
            batch_texts: list[str] = []

            for row in batch_rows:
                iid = str(row.get("IID", "")).strip()
                text = str(row.get("Text", "")).strip()
                if not iid:
                    continue
                if not text:
                    if args.skip_empty_text:
                        continue
                    text = "[EMPTY_TEXT]"
                batch_ids.append(iid)
                batch_texts.append(text)

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

    if not output_embeddings:
        raise RuntimeError("No embeddings were produced.")

    embeddings = torch.cat(output_embeddings, dim=0)
    iid_to_index = {iid: idx for idx, iid in enumerate(output_ids)}

    save_obj = {
        "ids": output_ids,
        "iid_to_index": iid_to_index,
        "embeddings": embeddings,
        "model_name": args.model_name,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "input_file": str(input_path.resolve()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_obj, output_path)
    output_r_path.parent.mkdir(parents=True, exist_ok=True)
    save_r_friendly_tsv_gz(output_r_path, output_ids, embeddings)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_r_path}")
    print(f"Number of persons: {len(output_ids)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
