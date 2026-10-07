import gc
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from svgpathtools import svgstr2paths

from starvector.data.util import clean_svg, rasterize_svg
from starvector.model.starvector_arch import StarVectorForCausalLM

MODEL_ID = "starvector/starvector-1b-im2svg"
INPUT = Path("assets/examples/sample-18.png")
OUT = Path("outputs/stage2-smoke/sample-18")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

print("[1/5] Loading StarVector-1B from local cache...", flush=True)
model = StarVectorForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    local_files_only=True,
).cuda().eval()
tokenizer = model.model.svg_transformer.tokenizer
source = Image.open(INPUT).convert("RGB")
image = model.process_images([source])[0].cuda()

print("[2/5] Deterministic single-candidate inference...", flush=True)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    raw_svg = model.generate_im2svg(
        {"image": image},
        max_length=7800,
        num_beams=1,
        do_sample=False,
    )[0]
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
generated_tokens = len(tokenizer.encode(raw_svg))
(OUT / "generated-raw.txt").write_text(raw_svg, encoding="utf-8")

print("[3/5] Validating and rasterizing SVG...", flush=True)
def xml_valid(text):
    try:
        ET.fromstring(text)
        return True
    except Exception:
        return False

def parser_valid(text):
    try:
        svgstr2paths(text)
        return True
    except Exception:
        return False

raw_xml_valid = xml_valid(raw_svg)
raw_parser_valid = parser_valid(raw_svg)
cleaned_svg = clean_svg(raw_svg)
clean_xml_valid = xml_valid(cleaned_svg)
clean_parser_valid = parser_valid(cleaned_svg)
(OUT / "generated.svg").write_text(cleaned_svg, encoding="utf-8")
rendered = rasterize_svg(cleaned_svg, resolution=512).convert("RGB")
reference = source.resize((512, 512), Image.Resampling.LANCZOS)
rendered.save(OUT / "generated-512.png")
reference.save(OUT / "reference-512.png")

# Release the 1B generation model before loading evaluation backbones.
del image, model
gc.collect()
torch.cuda.empty_cache()

print("[4/5] Computing official image metrics...", flush=True)
from starvector.metrics.compute_l2 import L2DistanceCalculator
from starvector.metrics.compute_SSIM import SSIMDistanceCalculator
from starvector.metrics.compute_LPIPS import LPIPSDistanceCalculator
from starvector.metrics.compute_dino_score import DINOScoreCalculator

batch = {"gt_im": [reference], "gen_im": [rendered]}

def scalar(result):
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)

mse = scalar(L2DistanceCalculator().calculate_score(batch, update=False))
ssim = scalar(SSIMDistanceCalculator().calculate_score(batch, update=False))
lpips_metric = LPIPSDistanceCalculator(device="cuda")
lpips_value = scalar(lpips_metric.calculate_score(batch, batch_size=1, update=False))
del lpips_metric
gc.collect()
torch.cuda.empty_cache()
dino_metric = DINOScoreCalculator(device="cuda")
dino = scalar(dino_metric.calculate_score(batch, update=False))

results = {
    "sample": str(INPUT),
    "model": MODEL_ID,
    "protocol": {
        "candidate_count": 1,
        "num_beams": 1,
        "do_sample": False,
        "max_length": 7800,
        "seed": 0,
        "raster_resolution": 512,
    },
    "validity": {
        "raw_xml_valid": raw_xml_valid,
        "raw_official_parser_valid": raw_parser_valid,
        "clean_xml_valid": clean_xml_valid,
        "clean_official_parser_valid": clean_parser_valid,
    },
    "generated_svg_tokens": generated_tokens,
    "ground_truth_svg_tokens": None,
    "ground_truth_svg_tokens_note": "Unavailable: sample-18.png has no paired ground-truth SVG in the repository.",
    "inference_seconds": elapsed,
    "peak_gpu_allocated_gib": peak_allocated,
    "peak_gpu_reserved_gib": peak_reserved,
    "metrics": {
        "mse_l2_official": mse,
        "ssim_official": ssim,
        "lpips_official": lpips_value,
        "dino_score_official": dino,
    },
}
(OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("[5/5] Done", flush=True)
print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
