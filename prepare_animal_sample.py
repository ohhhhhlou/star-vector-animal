import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops

from starvector.data.util import rasterize_svg


PARQUET_PATH = Path(
    "data/svg-animal-illustrations/data/"
    "train-00000-of-00001.parquet"
)

OUTPUT_ROOT = Path("outputs/animal-smoke")


def parse_viewbox(viewbox):
    """Parse an SVG viewBox into four floating-point values."""
    values = re.split(r"[\s,]+", viewbox.strip())

    if len(values) != 4:
        raise ValueError(
            f"Invalid viewBox: {viewbox!r}. "
            "Expected four values: min_x min_y width height."
        )

    min_x, min_y, width, height = map(float, values)

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid viewBox dimensions: width={width}, height={height}"
        )

    return min_x, min_y, width, height


def format_number(value):
    """Format coordinates without unnecessary trailing zeros."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def add_viewbox_padding(svg_code, padding_ratio=0.1):
    """
    Expand the SVG viewBox symmetrically.

    Example:
        viewBox="0 0 200 200"
    becomes:
        viewBox="-20 -20 240 240"
    when padding_ratio=0.1.
    """
    root = ET.fromstring(svg_code)

    original_viewbox = root.attrib.get("viewBox")

    if original_viewbox is None:
        raise ValueError("SVG does not contain a viewBox attribute.")

    min_x, min_y, width, height = parse_viewbox(original_viewbox)

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    new_min_x = min_x - pad_x
    new_min_y = min_y - pad_y
    new_width = width + 2 * pad_x
    new_height = height + 2 * pad_y

    normalized_viewbox = " ".join(
        [
            format_number(new_min_x),
            format_number(new_min_y),
            format_number(new_width),
            format_number(new_height),
        ]
    )

    # 只替换根SVG标签中的viewBox，尽量保持其他源码不变。
    pattern = re.compile(
        r'(<svg\b[^>]*\bviewBox\s*=\s*)(["\'])([^"\']+)(["\'])',
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_viewbox(match):
        quote = match.group(2)
        return f"{match.group(1)}{quote}{normalized_viewbox}{quote}"

    normalized_svg, replacement_count = pattern.subn(
        replace_viewbox,
        svg_code,
        count=1,
    )

    if replacement_count != 1:
        raise ValueError(
            "Could not replace the viewBox attribute in the SVG source."
        )

    # 再次验证修改后的SVG仍然是合法XML。
    ET.fromstring(normalized_svg)

    return normalized_svg, original_viewbox, normalized_viewbox


def check_non_blank(image):
    """Raise an error if the rendered image is completely white."""
    rgb_image = image.convert("RGB")
    white_background = Image.new(
        "RGB",
        rgb_image.size,
        color="white",
    )

    difference = ImageChops.difference(
        rgb_image,
        white_background,
    )

    if difference.getbbox() is None:
        raise RuntimeError(
            "The rendered image is completely white. "
            "The SVG may have failed to render."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract one Animal Illustrations SVG and render it "
            "with the official StarVector rasterizer."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset row index. Default: 0.",
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help=(
            "Padding ratio added around the SVG viewBox. "
            "Default: 0.1 (10 percent)."
        ),
    )

    args = parser.parse_args()

    if args.padding < 0:
        raise ValueError("--padding must be non-negative.")

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {PARQUET_PATH}"
        )

    df = pd.read_parquet(PARQUET_PATH)

    if not 0 <= args.index < len(df):
        raise IndexError(
            f"Index {args.index} is outside the dataset range "
            f"0-{len(df) - 1}."
        )

    sample = df.iloc[args.index]

    original_svg = sample["svg"]
    prompt = sample["prompt"]

    # 验证数据集中的原始SVG。
    ET.fromstring(original_svg)

    normalized_svg, original_viewbox, normalized_viewbox = (
        add_viewbox_padding(
            original_svg,
            padding_ratio=args.padding,
        )
    )

    sample_dir = OUTPUT_ROOT / f"{args.index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_svg_path = sample_dir / "original.svg"
    ground_truth_svg_path = sample_dir / "ground-truth.svg"
    ground_truth_png_path = sample_dir / "ground-truth.png"
    prompt_path = sample_dir / "prompt.txt"
    metadata_path = sample_dir / "metadata.json"

    original_svg_path.write_text(
        original_svg,
        encoding="utf-8",
    )

    ground_truth_svg_path.write_text(
        normalized_svg,
        encoding="utf-8",
    )

    prompt_path.write_text(
        prompt,
        encoding="utf-8",
    )

    # 使用StarVector官方渲染器。
    rendered_image = rasterize_svg(
        normalized_svg,
        resolution=224,
        dpi=128,
        scale=2,
    ).convert("RGB")

    if rendered_image.size != (224, 224):
        raise RuntimeError(
            f"Unexpected rendered size: {rendered_image.size}. "
            "Expected (224, 224)."
        )

    check_non_blank(rendered_image)

    rendered_image.save(
        ground_truth_png_path,
        format="PNG",
    )

    metadata = {
        "dataset_index": int(args.index),
        "prompt": prompt,
        "padding_ratio": args.padding,
        "original_viewbox": original_viewbox,
        "normalized_viewbox": normalized_viewbox,
        "original_svg_characters": len(original_svg),
        "normalized_svg_characters": len(normalized_svg),
        "render_width": rendered_image.width,
        "render_height": rendered_image.height,
        "image_mode": rendered_image.mode,
        "background": "white",
        "dpi": 128,
        "scale": 2,
        "renderer": "starvector.data.util.rasterize_svg",
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Dataset rows: {len(df)}")
    print(f"Dataset index: {args.index}")
    print(f"Prompt: {prompt}")
    print(f"Padding ratio: {args.padding}")
    print(f"Original viewBox: {original_viewbox}")
    print(f"Normalized viewBox: {normalized_viewbox}")
    print(f"Rendered image: {rendered_image.mode} {rendered_image.size}")
    print(f"Original SVG: {original_svg_path}")
    print(f"Ground-truth SVG: {ground_truth_svg_path}")
    print(f"Ground-truth PNG: {ground_truth_png_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()