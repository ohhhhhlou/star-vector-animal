#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}第{line_number}行解析失败: {error}"
                ) from error

            rows.append(row)

    return rows


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def match_categories(prompt, categories):
    """
    按完整英文单词匹配类别。

    例如：
    pony可以匹配“a blue pony”，
    但不会误匹配其他单词的一部分。
    """
    prompt = prompt.lower()
    matched = []

    for category in categories:
        pattern = rf"\b{re.escape(category.lower())}\b"

        if re.search(pattern, prompt):
            matched.append(category)

    return matched


def main():
    parser = argparse.ArgumentParser(
        description="为指定动物类别生成重采样训练集"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/animal-lora/train.jsonl"
        ),
        help="原始训练集JSONL",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/strength-animal"
        ),
        help="增强训练集输出目录",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["pony", "mouse"],
        help="需要加强的动物类别",
    )

    parser.add_argument(
        "--extra-copies",
        type=int,
        default=1,
        help="每个选中样本额外复制几次，默认1",
    )

    args = parser.parse_args()

    if args.extra_copies < 1:
        raise ValueError(
            "--extra-copies必须大于等于1"
        )

    if not args.input.is_file():
        raise FileNotFoundError(
            f"找不到原训练集：{args.input}"
        )

    original_rows = load_jsonl(args.input)

    selected_rows = []
    selected_ids = []
    category_counts = Counter()

    for row in original_rows:
        prompt = row.get("prompt", "")
        matched_categories = match_categories(
            prompt,
            args.categories,
        )

        if not matched_categories:
            continue

        selected_rows.append(row)
        selected_ids.append(str(row.get("id", "")))

        for category in matched_categories:
            category_counts[category] += 1

    # 保留全部原训练数据。
    augmented_rows = list(original_rows)

    # 选中的困难类别额外出现指定次数。
    for _ in range(args.extra_copies):
        augmented_rows.extend(selected_rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_jsonl = (
        args.output_dir
        / "train-strength.jsonl"
    )

    selected_ids_path = (
        args.output_dir
        / "resampled-ids.txt"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    write_jsonl(
        output_jsonl,
        augmented_rows,
    )

    with selected_ids_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for sample_id in selected_ids:
            file.write(f"{sample_id}\n")

    summary = {
        "source_file": str(args.input),
        "output_file": str(output_jsonl),
        "categories": args.categories,
        "extra_copies": args.extra_copies,
        "original_sample_count": len(original_rows),
        "selected_unique_sample_count": len(selected_rows),
        "extra_sample_count": (
            len(selected_rows)
            * args.extra_copies
        ),
        "augmented_sample_count": len(augmented_rows),
        "selected_count_by_category": {
            category: category_counts.get(category, 0)
            for category in args.categories
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=== 增强训练集生成完成 ===")
    print(f"原始训练样本: {len(original_rows)}")
    print(
        f"选中的不同样本: {len(selected_rows)}"
    )

    for category in args.categories:
        print(
            f"{category}样本: "
            f"{category_counts.get(category, 0)}"
        )

    print(
        f"每个选中样本额外出现: "
        f"{args.extra_copies}次"
    )

    print(
        f"额外训练记录: "
        f"{len(selected_rows) * args.extra_copies}"
    )

    print(
        f"增强后训练样本: "
        f"{len(augmented_rows)}"
    )

    print(f"增强训练集: {output_jsonl}")
    print(f"重采样ID: {selected_ids_path}")
    print(f"统计信息: {summary_path}")

    expected_count = (
        len(original_rows)
        + len(selected_rows) * args.extra_copies
    )

    if len(augmented_rows) != expected_count:
        raise RuntimeError(
            "增强训练集数量检查失败"
        )

    if not selected_rows:
        print(
            "警告：没有找到符合类别条件的训练样本。"
        )


if __name__ == "__main__":
    main()