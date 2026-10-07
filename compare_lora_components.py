#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from check_original_svg_components import (
    analyze_svg,
    make_component_preview,
)


def main():
    parser = argparse.ArgumentParser(
        description="对比原SVG和LoRA生成SVG的连通块数量"
    )

    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="测试集test.jsonl",
    )

    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="LoRA生成结果的samples目录",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/animal-eval/lora/component-comparison"
        ),
    )

    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--min-pixel-ratio",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    test_rows = []

    with args.jsonl.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            test_rows.append({
                "sample_id": str(item["id"]),
                "original_svg": Path(item["svg"]),
            })

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir = (
        args.output_dir / "mismatch-previews"
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for index, item in enumerate(test_rows, start=1):
        sample_id = item["sample_id"]
        original_path = item["original_svg"]

        generated_path = (
            args.generated_dir
            / sample_id
            / "generated.svg"
        )

        row = {
            "sample_id": sample_id,
            "original_svg": str(original_path),
            "generated_svg": str(generated_path),
            "original_exists": original_path.is_file(),
            "generated_exists": generated_path.is_file(),
            "comparison_success": False,
            "error": "",
            "original_component_count": "",
            "generated_component_count": "",
            "component_difference": "",
            "status": "",
            "original_largest_component_ratio": "",
            "generated_largest_component_ratio": "",
            "largest_component_ratio_difference": "",
            "original_component_sizes": "",
            "generated_component_sizes": "",
        }

        if not original_path.is_file():
            row["error"] = "原SVG不存在"
            results.append(row)
            print(f"{sample_id}: 原SVG不存在")
            continue

        if not generated_path.is_file():
            row["error"] = "生成SVG不存在"
            results.append(row)
            print(f"{sample_id}: 生成SVG不存在")
            continue

        try:
            original = analyze_svg(
                svg_path=original_path,
                render_size=args.render_size,
                alpha_threshold=args.alpha_threshold,
                min_pixel_ratio=args.min_pixel_ratio,
                min_pixels=args.min_pixels,
            )

            generated = analyze_svg(
                svg_path=generated_path,
                render_size=args.render_size,
                alpha_threshold=args.alpha_threshold,
                min_pixel_ratio=args.min_pixel_ratio,
                min_pixels=args.min_pixels,
            )

            original_count = original["component_count"]
            generated_count = generated["component_count"]

            difference = generated_count - original_count

            if difference > 0:
                status = "生成结果更加分散"
            elif difference < 0:
                status = "生成结果连通块更少"
            else:
                status = "数量一致"

            original_largest = (
                original["largest_component_ratio"]
            )

            generated_largest = (
                generated["largest_component_ratio"]
            )

            row.update({
                "comparison_success": True,
                "original_component_count": original_count,
                "generated_component_count": generated_count,
                "component_difference": difference,
                "status": status,
                "original_largest_component_ratio": (
                    original_largest
                ),
                "generated_largest_component_ratio": (
                    generated_largest
                ),
                "largest_component_ratio_difference": (
                    generated_largest - original_largest
                ),
                "original_component_sizes": (
                    original["component_sizes"]
                ),
                "generated_component_sizes": (
                    generated["component_sizes"]
                ),
            })

            print(
                f"{sample_id}: "
                f"原={original_count}, "
                f"生成={generated_count}, "
                f"差值={difference:+d}, "
                f"{status}"
            )

            # 只为数量不一致的样本保存彩色检查图。
            if difference != 0:
                original_preview = make_component_preview(
                    original["labels"],
                    original["rgba"],
                )

                generated_preview = make_component_preview(
                    generated["labels"],
                    generated["rgba"],
                )

                original_preview.save(
                    preview_dir
                    / f"{sample_id}-original.png"
                )

                generated_preview.save(
                    preview_dir
                    / f"{sample_id}-generated.png"
                )

        except Exception as error:
            row["error"] = str(error)

            print(
                f"{sample_id}: 检查失败，{error}"
            )

        results.append(row)

        if index % 20 == 0 or index == len(test_rows):
            print(
                f"处理进度: {index}/{len(test_rows)}"
            )

    report_path = (
        args.output_dir
        / "component-comparison.csv"
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    successful = [
        row for row in results
        if row["comparison_success"]
    ]

    same = [
        row for row in successful
        if row["component_difference"] == 0
    ]

    more = [
        row for row in successful
        if row["component_difference"] > 0
    ]

    fewer = [
        row for row in successful
        if row["component_difference"] < 0
    ]

    failed = [
        row for row in results
        if not row["comparison_success"]
    ]

    # 计算每个样本的有符号差值和绝对差值。
    differences = [
        int(row["component_difference"])
        for row in successful
    ]

    absolute_differences = [
        abs(difference)
        for difference in differences
    ]

    # 筛选绝对差值大于等于3的样本。
    large_difference_rows = [
        row for row in successful
        if abs(int(row["component_difference"])) >= 3
    ]

    # 按差异大小从大到小排列。
    large_difference_rows.sort(
        key=lambda row: abs(
            int(row["component_difference"])
        ),
        reverse=True,
    )

    large_difference_path = (
        args.output_dir
        / "large-component-differences.csv"
    )

    if large_difference_rows:
        with large_difference_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    large_difference_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(large_difference_rows)

    print("\n=== 连通块对比完成 ===")
    print(f"测试样本: {len(test_rows)}")
    print(f"成功比较: {len(successful)}")
    print(f"比较失败: {len(failed)}")
    print(f"数量一致: {len(same)}")
    print(f"生成结果更加分散: {len(more)}")
    print(f"生成结果连通块更少: {len(fewer)}")

    if successful:
        exact_rate = len(same) / len(successful)

        total_signed_difference = sum(differences)

        average_signed_difference = (
            total_signed_difference / len(successful)
        )

        total_absolute_difference = sum(
            absolute_differences
        )

        average_absolute_difference = (
            total_absolute_difference
            / len(successful)
        )

        print(
            f"连通块数量一致率: {exact_rate:.2%}"
        )

        print(
            f"总有符号差值: "
            f"{total_signed_difference:+d}"
        )

        print(
            f"平均有符号差值: "
            f"{average_signed_difference:+.3f}"
        )

        print(
            f"总绝对差值: "
            f"{total_absolute_difference}"
        )

        print(
            f"平均绝对差值: "
            f"{average_absolute_difference:.3f}"
        )

    print(
        f"绝对差值大于等于3的样本: "
        f"{len(large_difference_rows)}"
    )

    print(f"CSV报告: {report_path}")
    print(f"不一致样本检查图: {preview_dir}")

    if large_difference_rows:
        print(
            f"大差异样本报告: "
            f"{large_difference_path}"
        )

        print("\n绝对差值大于等于3的样本:")

        for row in large_difference_rows:
            difference = int(
                row["component_difference"]
            )

            print(
                row["sample_id"],
                f"原={row['original_component_count']}",
                f"生成={row['generated_component_count']}",
                f"差值={difference:+d}",
                f"绝对差值={abs(difference)}",
            )

if __name__ == "__main__":
    main()