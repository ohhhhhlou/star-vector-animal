#!/usr/bin/env python3

import argparse
import csv
import io
import math
import json
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from scipy import ndimage


def render_svg_in_memory(svg_path, render_size):
    """
    将SVG渲染到内存，不在硬盘保存中间PNG。
    返回RGBA像素数组。
    """
    svg_text = svg_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=render_size,
        output_height=render_size,
    )

    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return np.asarray(image)


def create_foreground_mask(rgba, alpha_threshold):
    """
    透明底SVG：
    alpha大于阈值的像素视为前景。
    """
    alpha = rgba[:, :, 3]
    return alpha > alpha_threshold


def find_components(mask, min_pixel_ratio, min_pixels):
    """
    使用八方向连接计算连通块。

    太小的区域视为噪点，不计入最终数量。
    """
    structure = np.ones((3, 3), dtype=np.uint8)

    raw_labels, raw_count = ndimage.label(
        mask,
        structure=structure,
    )

    foreground_pixels = int(mask.sum())

    calculated_threshold = math.ceil(
        foreground_pixels * min_pixel_ratio
    )

    effective_min_pixels = max(
        min_pixels,
        calculated_threshold,
    )

    components = []

    for label_id in range(1, raw_count + 1):
        component_size = int(
            np.sum(raw_labels == label_id)
        )

        if component_size >= effective_min_pixels:
            components.append({
                "old_label": label_id,
                "size": component_size,
            })

    # 面积从大到小排列。
    components.sort(
        key=lambda item: item["size"],
        reverse=True,
    )

    filtered_labels = np.zeros_like(
        raw_labels,
        dtype=np.int32,
    )

    for new_label, component in enumerate(
        components,
        start=1,
    ):
        filtered_labels[
            raw_labels == component["old_label"]
        ] = new_label

        component["new_label"] = new_label

    return {
        "raw_component_count": int(raw_count),
        "component_count": len(components),
        "foreground_pixels": foreground_pixels,
        "minimum_component_pixels": effective_min_pixels,
        "components": components,
        "labels": filtered_labels,
    }


def make_component_preview(labels, rgba):
    """
    为不同连通块分配不同颜色。
    灰色部分表示被过滤掉的小区域。
    """
    height, width = labels.shape

    preview = np.full(
        (height, width, 4),
        255,
        dtype=np.uint8,
    )

    colors = [
        (230, 57, 70),
        (29, 185, 84),
        (0, 120, 255),
        (255, 166, 0),
        (142, 68, 173),
        (0, 180, 180),
        (255, 105, 180),
        (120, 90, 60),
        (100, 149, 237),
        (80, 160, 80),
    ]

    for label_id in range(1, int(labels.max()) + 1):
        color = colors[(label_id - 1) % len(colors)]
        component_mask = labels == label_id

        preview[component_mask, 0] = color[0]
        preview[component_mask, 1] = color[1]
        preview[component_mask, 2] = color[2]
        preview[component_mask, 3] = 255

    # 给前景轮廓加一层黑边，方便查看相邻区域。
    foreground = labels > 0
    eroded = ndimage.binary_erosion(
        foreground,
        structure=np.ones((3, 3), dtype=bool),
    )
    edge = foreground & ~eroded

    preview[edge, 0:3] = 0
    preview[edge, 3] = 255

    return Image.fromarray(preview, mode="RGBA")


def analyze_svg(
    svg_path,
    render_size,
    alpha_threshold,
    min_pixel_ratio,
    min_pixels,
):
    rgba = render_svg_in_memory(
        svg_path,
        render_size,
    )

    mask = create_foreground_mask(
        rgba,
        alpha_threshold,
    )

    result = find_components(
        mask,
        min_pixel_ratio,
        min_pixels,
    )

    foreground_pixels = result["foreground_pixels"]
    image_pixels = render_size * render_size

    foreground_ratio = (
        foreground_pixels / image_pixels
        if image_pixels > 0
        else 0.0
    )

    component_sizes = [
        component["size"]
        for component in result["components"]
    ]

    largest_component_ratio = 0.0

    if foreground_pixels > 0 and component_sizes:
        largest_component_ratio = (
            component_sizes[0] / foreground_pixels
        )

    return {
        "sample_id": svg_path.stem,
        "svg_path": str(svg_path),
        "render_success": True,
        "error": "",
        "render_size": render_size,
        "foreground_pixels": foreground_pixels,
        "foreground_ratio": foreground_ratio,
        "raw_component_count": result["raw_component_count"],
        "component_count": result["component_count"],
        "minimum_component_pixels": result[
            "minimum_component_pixels"
        ],
        "largest_component_ratio": largest_component_ratio,
        "component_sizes": ";".join(
            str(size) for size in component_sizes
        ),
        "labels": result["labels"],
        "rgba": rgba,
    }


def main():
    parser = argparse.ArgumentParser(
        description="检查测试集原SVG的连通块数量"
    )

    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="测试集test.jsonl路径",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/test-original-component-check"
        ),
        help="报告和检查图输出目录",
    )

    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="统一渲染尺寸，默认512",
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=16,
        help="透明度阈值，默认16",
    )

    parser.add_argument(
        "--min-pixel-ratio",
        type=float,
        default=0.001,
        help="最小连通块占前景比例，默认0.001",
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
        help="连通块至少包含的像素数，默认10",
    )

    parser.add_argument(
        "--preview-limit",
        type=int,
        default=30,
        help="保存多少张彩色检查图；0表示全部保存",
    )

    args = parser.parse_args()

    # 读取test.jsonl中指定的142个测试样本。
    test_rows = []

    with args.jsonl.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL第{line_number}行解析失败: {error}"
                ) from error

            if "id" not in item or "svg" not in item:
                raise ValueError(
                    f"JSONL第{line_number}行缺少id或svg字段"
                )

            sample_id = str(item["id"])
            svg_path = Path(item["svg"])

            test_rows.append({
                "sample_id": sample_id,
                "svg_path": svg_path,
            })

    if not test_rows:
        raise SystemExit(
            f"测试集没有样本：{args.jsonl}"
        )

    print(f"测试集样本数: {len(test_rows)}")
    print(f"渲染尺寸: {args.render_size}×{args.render_size}")
    print("连接规则: 八方向连接")
    print(
        f"最小区域: 前景面积的"
        f"{args.min_pixel_ratio * 100:.3f}%"
    )

    missing_files = [
        row for row in test_rows
        if not row["svg_path"].is_file()
    ]

    if missing_files:
        print("\n以下SVG文件不存在（最多显示10个）:")

        for row in missing_files[:10]:
            print(
                row["sample_id"],
                row["svg_path"],
            )

        raise SystemExit(
            f"共有{len(missing_files)}个SVG文件不存在。"
            "请确认当前目录是star-vector项目根目录。"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir = (
        args.output_dir / "component-previews"
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index, test_item in enumerate(
        test_rows,
        start=1,
    ):
        sample_id = test_item["sample_id"]
        svg_path = test_item["svg_path"]

        try:
            result = analyze_svg(
                svg_path=svg_path,
                render_size=args.render_size,
                alpha_threshold=args.alpha_threshold,
                min_pixel_ratio=args.min_pixel_ratio,
                min_pixels=args.min_pixels,
            )

            # analyze_svg原来会使用文件名target作为ID，
            # 这里改成测试集真正的编号，例如000762。
            result["sample_id"] = sample_id
            result["svg_path"] = str(svg_path)

            should_save_preview = (
                args.preview_limit == 0
                or index <= args.preview_limit
            )

            if should_save_preview:
                preview = make_component_preview(
                    result["labels"],
                    result["rgba"],
                )

                preview_path = (
                    preview_dir
                    / f"{sample_id}-components.png"
                )

                preview.save(preview_path)

            row = {
                key: value
                for key, value in result.items()
                if key not in {"labels", "rgba"}
            }

            print(
                f"{sample_id}: "
                f"连通块={row['component_count']}, "
                f"过滤前={row['raw_component_count']}, "
                f"最大块占比="
                f"{row['largest_component_ratio']:.3f}"
            )

        except Exception as error:
            row = {
                "sample_id": sample_id,
                "svg_path": str(svg_path),
                "render_success": False,
                "error": str(error),
                "render_size": args.render_size,
                "foreground_pixels": "",
                "foreground_ratio": "",
                "raw_component_count": "",
                "component_count": "",
                "minimum_component_pixels": "",
                "largest_component_ratio": "",
                "component_sizes": "",
            }

            print(
                f"{sample_id}: 检查失败，{error}"
            )

        rows.append(row)

        if index % 20 == 0 or index == len(test_rows):
            print(
                f"处理进度: {index}/{len(test_rows)}"
            )

    report_path = (
        args.output_dir
        / "original-component-report.csv"
    )

    with report_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = [
        row for row in rows
        if row["render_success"]
    ]

    failed = [
        row for row in rows
        if not row["render_success"]
    ]

    component_counts = [
        row["component_count"]
        for row in successful
    ]

    print("\n=== 测试集原SVG检查完成 ===")
    print(f"总样本数: {len(test_rows)}")
    print(f"成功渲染: {len(successful)}")
    print(f"渲染失败: {len(failed)}")
    print(f"CSV报告: {report_path}")
    print(f"彩色检查图: {preview_dir}")

    if component_counts:
        print(
            "平均连通块数量: "
            f"{np.mean(component_counts):.2f}"
        )
        print(
            "连通块数量中位数: "
            f"{np.median(component_counts):.2f}"
        )
        print(
            "最大连通块数量: "
            f"{max(component_counts)}"
        )

    print("\n连通块数量最多的样本（前10个）:")

    sorted_rows = sorted(
        successful,
        key=lambda row: row["component_count"],
        reverse=True,
    )

    for row in sorted_rows[:10]:
        print(
            row["sample_id"],
            f"连通块={row['component_count']}",
            f"最大块占比="
            f"{row['largest_component_ratio']:.3f}",
            f"各块面积={row['component_sizes']}",
        )


if __name__ == "__main__":
    main()