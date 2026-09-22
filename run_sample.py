from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from starvector.data.util import process_and_rasterize_svg


MODEL_ID = "starvector/starvector-1b-im2svg"
INPUT_IMAGE = Path("assets/examples/sample-18.png")
OUTPUT_DIR = Path("outputs/stage1")

RAW_OUTPUT_PATH = OUTPUT_DIR / "sample-18-raw.txt"
SVG_OUTPUT_PATH = OUTPUT_DIR / "sample-18-generated.svg"
PNG_OUTPUT_PATH = OUTPUT_DIR / "sample-18-generated.png"


def main() -> None:
    """Run the official StarVector-1B Image-to-SVG example."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; StarVector inference requires a GPU.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the already downloaded official StarVector-1B checkpoint.
    model_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_files_only=True,
    )
    print(f"StarVector checkpoint: {model_dir}")

    print("Loading StarVector-1B...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = model.model.processor
    model.cuda()
    model.eval()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input image: {INPUT_IMAGE}")

    image_pil = Image.open(INPUT_IMAGE).convert("RGB")
    image = processor(
        image_pil,
        return_tensors="pt",
    )["pixel_values"].cuda()

    # Keep the tensor shape consistent with the official README example.
    if image.shape[0] != 1:
        image = image.squeeze(0)

    print("Generating SVG...")
    with torch.inference_mode():
        raw_svg = model.generate_im2svg(
            {"image": image},
            max_length=4000,
        )[0]

    # Preserve the original model response for later debugging.
    RAW_OUTPUT_PATH.write_text(raw_svg, encoding="utf-8")

    # Clean the SVG and rasterize it to verify that a renderer accepts it.
    svg, raster_image = process_and_rasterize_svg(raw_svg)
    SVG_OUTPUT_PATH.write_text(svg, encoding="utf-8")
    raster_image.save(PNG_OUTPUT_PATH)

    print(f"Raw output: {RAW_OUTPUT_PATH}")
    print(f"SVG output: {SVG_OUTPUT_PATH}")
    print(f"PNG preview: {PNG_OUTPUT_PATH}")
    print(f"SVG characters: {len(svg)}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
