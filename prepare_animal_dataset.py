import argparse
import json
import random
import traceback
from pathlib import Path

import pandas as pd

from prepare_animal_sample import (
    PARQUET_PATH,
    add_viewbox_padding,
    check_non_blank,
)
from starvector.data.util import rasterize_svg


OUTPUT_ROOT = Path("data/animal-lora")


def process_sample(index, sample, output_root, padding):
    original_svg = str(sample["svg"])
    prompt = str(sample["prompt"])

    normalized_svg, original_viewbox, normalized_viewbox = (
        add_viewbox_padding(
            original_svg,
            padding_ratio=padding,
        )
    )

    sample_dir = output_root / "samples" / f"{index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = rasterize_svg(
        normalized_svg,
        resolution=224,
        dpi=128,
        scale=2,
    ).convert("RGB")

    if image.size != (224, 224):
        raise RuntimeError(f"Unexpected image size: {image.size}")

    check_non_blank(image)

    image_path = sample_dir / "image.png"
    target_svg_path = sample_dir / "target.svg"
    prompt_path = sample_dir / "prompt.txt"

    image.save(image_path, format="PNG")
    target_svg_path.write_text(normalized_svg, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    return {
        "id": f"{index:06d}",
        "dataset_index": index,
        "image": str(image_path),
        "svg": str(target_svg_path),
        "prompt": prompt,
        "original_viewbox": original_viewbox,
        "normalized_viewbox": normalized_viewbox,
        "padding_ratio": padding,
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows for a smoke test.",
    )
    args = parser.parse_args()

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(PARQUET_PATH)

    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.validation_ratio + args.test_ratio >= 1:
        raise ValueError(
            "validation-ratio + test-ratio must be less than 1"
        )

    dataframe = pd.read_parquet(PARQUET_PATH)
    required_columns = {"prompt", "svg"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if args.limit is not None:
        dataframe = dataframe.iloc[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    successful = []
    failures = []

    for position, (index, sample) in enumerate(
        dataframe.iterrows(), start=1
    ):
        try:
            record = process_sample(
                index=int(index),
                sample=sample,
                output_root=args.output_root,
                padding=args.padding,
            )
            successful.append(record)
            status = "OK"
        except Exception as error:
            failures.append(
                {
                    "dataset_index": int(index),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            status = f"FAILED: {error}"

        print(
            f"[{position}/{len(dataframe)}] index={index}: {status}",
            flush=True,
        )

    random_generator = random.Random(args.seed)
    random_generator.shuffle(successful)

    test_count = round(len(successful) * args.test_ratio)
    validation_count = round(
        len(successful) * args.validation_ratio
    )

    test_records = successful[:test_count]
    validation_records = successful[
        test_count:test_count + validation_count
    ]
    train_records = successful[test_count + validation_count:]

    write_jsonl(args.output_root / "train.jsonl", train_records)
    write_jsonl(
        args.output_root / "validation.jsonl",
        validation_records,
    )
    write_jsonl(args.output_root / "test.jsonl", test_records)
    write_jsonl(args.output_root / "failures.jsonl", failures)

    summary = {
        "source": str(PARQUET_PATH),
        "total_attempted": len(dataframe),
        "successful": len(successful),
        "failed": len(failures),
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_samples": len(test_records),
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "padding_ratio": args.padding,
        "resolution": 224,
    }

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
