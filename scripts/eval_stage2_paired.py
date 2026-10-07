import gc
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from svgpathtools import svgstr2paths

from starvector.data.util import clean_svg, rasterize_svg
from starvector.model.starvector_arch import StarVectorForCausalLM

MODEL_ID = "starvector/starvector-1b-im2svg"
GT_SVG_PATH = Path("data/svg-animal-illustrations/samples/sample_bird.svg")
OUT = Path("outputs/stage2-paired/sample_bird")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

gt_svg = GT_SVG_PATH.read_text(encoding="utf-8")
source = rasterize_svg(gt_svg, resolution=224).convert("RGB")
source.save(OUT / "input-from-ground-truth.png")
(OUT / "ground-truth.svg").write_text(gt_svg, encoding="utf-8")

print("[1/5] Loading StarVector-1B from local cache...", flush=True)
model = StarVectorForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, local_files_only=True
).cuda().eval()
tokenizer = model.model.svg_transformer.tokenizer
gt_tokens = len(tokenizer.encode(gt_svg))
image = model.process_images([source])[0].cuda()

print("[2/5] Deterministic paired-sample inference...", flush=True)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    raw_svg = model.generate_im2svg(
        {"image": image}, max_length=7800, num_beams=1, do_sample=False
    )[0]
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
generated_tokens = len(tokenizer.encode(raw_svg))
(OUT / "generated-raw.txt").write_text(raw_svg, encoding="utf-8")

print("[3/5] Validating and rasterizing SVG pair...", flush=True)
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

cleaned_svg = clean_svg(raw_svg)
(OUT / "generated.svg").write_text(cleaned_svg, encoding="utf-8")
generated_image = rasterize_svg(cleaned_svg, resolution=512).convert("RGB")
ground_truth_image = rasterize_svg(gt_svg, resolution=512).convert("RGB")
generated_image.save(OUT / "generated-512.png")
ground_truth_image.save(OUT / "ground-truth-512.png")

validity = {
    "ground_truth_xml_valid": xml_valid(gt_svg),
    "ground_truth_official_parser_valid": parser_valid(gt_svg),
    "raw_generated_xml_valid": xml_valid(raw_svg),
    "raw_generated_official_parser_valid": parser_valid(raw_svg),
    "clean_generated_xml_valid": xml_valid(cleaned_svg),
    "clean_generated_official_parser_valid": parser_valid(cleaned_svg),
}

del image, model
gc.collect()
torch.cuda.empty_cache()

print("[4/5] Computing official MSE, SSIM, LPIPS and DinoScore...", flush=True)
from starvector.metrics.compute_l2 import L2DistanceCalculator
from starvector.metrics.compute_SSIM import SSIMDistanceCalculator
from starvector.metrics.compute_LPIPS import LPIPSDistanceCalculator
from starvector.metrics.compute_dino_score import DINOScoreCalculator

batch = {"gt_im": [ground_truth_image], "gen_im": [generated_image]}
def scalar(result):
    return float(result[0] if isinstance(result, tuple) else result)

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
    "sample": str(GT_SVG_PATH),
    "paired_sample": True,
    "model": MODEL_ID,
    "protocol": {
        "candidate_count": 1, "num_beams": 1, "do_sample": False,
        "max_length": 7800, "seed": 0, "raster_resolution": 512
    },
    "validity": validity,
    "ground_truth_svg_tokens": gt_tokens,
    "generated_svg_tokens": generated_tokens,
    "token_length_ratio_generated_over_ground_truth": generated_tokens / gt_tokens,
    "inference_seconds": elapsed,
    "peak_gpu_allocated_gib": peak_allocated,
    "peak_gpu_reserved_gib": peak_reserved,
    "metrics": {
        "mse_l2_official": mse,
        "ssim_official": ssim,
        "lpips_official": lpips_value,
        "dino_score_official": dino
    }
}
(OUT / "results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("[5/5] Done", flush=True)
print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
