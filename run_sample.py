from pathlib import Path

import torch
from PIL import Image

from starvector.data.util import process_and_rasterize_svg
from starvector.model.starvector_arch import StarVectorForCausalLM

MODEL_ID = "starvector/starvector-1b-im2svg"
INPUT_IMAGE = Path("outputs/animal-smoke/0000/ground-truth.png")
OUTPUT_DIR = Path("outputs/animal-smoke/0000")
RAW_OUTPUT_PATH = OUTPUT_DIR / "generate.txt"
SVG_OUTPUT_PATH = OUTPUT_DIR / "generated.svg"
PNG_OUTPUT_PATH = OUTPUT_DIR / "generated.png"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; StarVector inference requires a GPU.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading StarVector-1B from the local Hugging Face cache...")
    starvector = StarVectorForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    starvector.cuda()
    starvector.eval()

    print(f"Model class: {type(starvector).__module__}.{type(starvector).__name__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input image: {INPUT_IMAGE}")

    image_pil = Image.open(INPUT_IMAGE).convert("RGB")
    image = starvector.process_images([image_pil])[0].cuda()
    print(f"Processed image shape: {tuple(image.shape)}")

    print("Generating SVG...")
    with torch.inference_mode():
        raw_svg = starvector.generate_im2svg(
            {"image": image},
            max_length=6000,
        )[0]

    RAW_OUTPUT_PATH.write_text(raw_svg, encoding="utf-8")
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
