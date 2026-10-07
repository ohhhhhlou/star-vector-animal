import argparse
import gc
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from svgpathtools import svgstr2paths
from starvector.data.util import clean_svg, rasterize_svg
from starvector.model.starvector_arch import StarVectorForCausalLM


# Official StarVector-1B im2svg generation configuration:
# configs/generation/hf/starvector-1b/im2svg.yaml
OFFICIAL_GENERATION_PARAMS = {
    "max_length": 7800,
    "min_length": 10,
    "num_beams": 1,
    "temperature": 0.2,
    "generation_sweep": False,
    "num_captions": 1,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "top_p": 0.95,
    "do_sample": True,
    "use_nucleus_sampling": True,
    "logit_bias": 5,
    "stream": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one SVG-Diagrams sample with official generation settings."
    )
    parser.add_argument(
        "--parquet",
        default="data/SVG-test/test-00000-of-00001.parquet",
        help="SVG-Diagrams test Parquet file.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--model", default="starvector/starvector-1b-im2svg"
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage2-svg-diagrams"
    )
    parser.add_argument(
        "--skip-perceptual-metrics",
        action="store_true",
        help="Skip LPIPS and DinoScore for quick debugging.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_xml(svg_text):
    try:
        ET.fromstring(svg_text)
        return True, None
    except Exception as error:
        return False, str(error)


def check_official_parser(svg_text):
    try:
        svgstr2paths(svg_text)
        return True, None
    except Exception as error:
        return False, str(error)


def unpack_metric(result):
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def main():
    args = parse_args()
    seed = 0
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path.resolve()}")

    print("[1/7] Reading SVG-Diagrams test set", flush=True)
    dataframe = pd.read_parquet(parquet_path)
    required_columns = {"Filename", "Svg"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}; "
            f"available: {list(dataframe.columns)}"
        )
    if not 0 <= args.index < len(dataframe):
        raise IndexError(
            f"index={args.index} out of range 0..{len(dataframe) - 1}"
        )

    row = dataframe.iloc[args.index]
    filename = str(row["Filename"])
    ground_truth_svg = str(row["Svg"])
    sample_name = Path(filename).stem
    output_dir = Path(args.output_dir) / f"{args.index:04d}-{sample_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ground-truth.svg").write_text(
        ground_truth_svg, encoding="utf-8"
    )

    print("Dataset size:", len(dataframe))
    print("Sample index:", args.index)
    print("Filename:", filename)
    print("Ground-truth SVG characters:", len(ground_truth_svg))
    print("Output directory:", output_dir)

    print("[2/7] Rasterizing ground-truth SVG", flush=True)
    input_image = rasterize_svg(
        ground_truth_svg, resolution=224
    ).convert("RGB")
    ground_truth_512 = rasterize_svg(
        ground_truth_svg, resolution=512
    ).convert("RGB")
    input_image.save(output_dir / "input-224.png")
    ground_truth_512.save(output_dir / "ground-truth-512.png")

    print("[3/7] Loading StarVector-1B", flush=True)
    model = StarVectorForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).cuda().eval()
    tokenizer = model.model.svg_transformer.tokenizer
    ground_truth_tokens = len(
        tokenizer.encode(ground_truth_svg, add_special_tokens=False)
    )
    processed_image = model.process_images([input_image])[0].cuda()
    print("Ground-truth SVG tokens:", ground_truth_tokens)
    print("Model input tensor:", tuple(processed_image.shape))

    print("[4/7] Generating one candidate with official settings", flush=True)
    print(
        json.dumps(
            OFFICIAL_GENERATION_PARAMS, indent=2, ensure_ascii=False
        )
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_raw = model.generate_im2svg(
            {"image": processed_image},
            **OFFICIAL_GENERATION_PARAMS,
        )[0]

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start_time
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    generated_tokens = len(
        tokenizer.encode(generated_raw, add_special_tokens=False)
    )
    has_svg_start_tag = "<svg" in generated_raw
    has_svg_end_tag = "</svg>" in generated_raw
    likely_hit_max_length = (
        generated_tokens >= OFFICIAL_GENERATION_PARAMS["max_length"] - 300
        and not has_svg_end_tag
    )
    (output_dir / "generated-raw.txt").write_text(
        generated_raw, encoding="utf-8"
    )

    print("Inference seconds:", round(inference_seconds, 3))
    print("Generated tokens:", generated_tokens)
    print("Has <svg>:", has_svg_start_tag)
    print("Has </svg>:", has_svg_end_tag)
    print("Likely hit max_length:", likely_hit_max_length)
    print("Peak GPU allocated:", round(peak_allocated_gib, 3), "GiB")
    print("Peak GPU reserved:", round(peak_reserved_gib, 3), "GiB")

    print("[5/7] Validating and cleaning generated SVG", flush=True)
    raw_xml_valid, raw_xml_error = check_xml(generated_raw)
    raw_parser_valid, raw_parser_error = check_official_parser(
        generated_raw
    )
    try:
        generated_svg = clean_svg(generated_raw)
        clean_svg_error = None
    except Exception as error:
        generated_svg = generated_raw
        clean_svg_error = str(error)

    clean_xml_valid, clean_xml_error = check_xml(generated_svg)
    clean_parser_valid, clean_parser_error = check_official_parser(
        generated_svg
    )
    (output_dir / "generated.svg").write_text(
        generated_svg, encoding="utf-8"
    )

    validity = {
        "raw_xml_valid": raw_xml_valid,
        "raw_xml_error": raw_xml_error,
        "raw_official_parser_valid": raw_parser_valid,
        "raw_official_parser_error": raw_parser_error,
        "clean_svg_error": clean_svg_error,
        "clean_xml_valid": clean_xml_valid,
        "clean_xml_error": clean_xml_error,
        "clean_official_parser_valid": clean_parser_valid,
        "clean_official_parser_error": clean_parser_error,
    }
    print(json.dumps(validity, indent=2, ensure_ascii=False))

    generation_status = {
        "has_svg_start_tag": has_svg_start_tag,
        "has_svg_end_tag": has_svg_end_tag,
        "likely_hit_max_length": likely_hit_max_length,
    }

    if not clean_xml_valid:
        failure_results = {
            "dataset": "starvector/svg-diagrams",
            "split": "test",
            "sample_index": args.index,
            "filename": filename,
            "model": args.model,
            "official_generation_params": OFFICIAL_GENERATION_PARAMS,
            "generation_status": generation_status,
            "validity": validity,
            "error": "Cleaned generated SVG is invalid; metrics skipped.",
        }
        (output_dir / "results.json").write_text(
            json.dumps(failure_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise RuntimeError(
            "Generated SVG remains invalid; inspect generated-raw.txt"
        )

    print("[6/7] Rasterizing generated SVG", flush=True)
    generated_512 = rasterize_svg(
        generated_svg, resolution=512
    ).convert("RGB")
    generated_512.save(output_dir / "generated-512.png")

    del processed_image
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("[7/7] Computing official metrics", flush=True)
    from starvector.metrics.compute_l2 import L2DistanceCalculator
    from starvector.metrics.compute_SSIM import SSIMDistanceCalculator

    metric_batch = {
        "gt_im": [ground_truth_512],
        "gen_im": [generated_512],
    }
    mse = unpack_metric(
        L2DistanceCalculator().calculate_score(
            metric_batch, update=False
        )
    )
    ssim = unpack_metric(
        SSIMDistanceCalculator().calculate_score(
            metric_batch, update=False
        )
    )

    lpips_score = None
    dino_score = None
    if not args.skip_perceptual_metrics:
        print("Computing LPIPS...", flush=True)
        from starvector.metrics.compute_LPIPS import LPIPSDistanceCalculator

        lpips_metric = LPIPSDistanceCalculator(device="cuda")
        lpips_score = unpack_metric(
            lpips_metric.calculate_score(
                metric_batch, batch_size=1, update=False
            )
        )
        del lpips_metric
        gc.collect()
        torch.cuda.empty_cache()

        print("Computing DinoScore...", flush=True)
        from starvector.metrics.compute_dino_score import DINOScoreCalculator

        dino_metric = DINOScoreCalculator(device="cuda")
        dino_score = unpack_metric(
            dino_metric.calculate_score(
                metric_batch, update=False
            )
        )
        del dino_metric
        gc.collect()
        torch.cuda.empty_cache()

    results = {
        "dataset": "starvector/svg-diagrams",
        "split": "test",
        "dataset_size": len(dataframe),
        "sample_index": args.index,
        "filename": filename,
        "model": args.model,
        "protocol": {
            "seed": seed,
            "candidate_count": 1,
            **OFFICIAL_GENERATION_PARAMS,
            "model_input_resolution": 224,
            "metric_resolution": 512,
        },
        "generation_status": generation_status,
        "validity": validity,
        "ground_truth_svg_characters": len(ground_truth_svg),
        "generated_svg_characters": len(generated_svg),
        "ground_truth_svg_tokens": ground_truth_tokens,
        "generated_svg_tokens": generated_tokens,
        "token_length_ratio": (
            generated_tokens / ground_truth_tokens
            if ground_truth_tokens
            else None
        ),
        "inference_seconds": inference_seconds,
        "peak_gpu_allocated_gib": peak_allocated_gib,
        "peak_gpu_reserved_gib": peak_reserved_gib,
        "metrics": {
            "mse_l2_official": mse,
            "ssim_official": ssim,
            "lpips_official": lpips_score,
            "dino_score_official": dino_score,
        },
    }

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("Evaluation complete:", results_path)


if __name__ == "__main__":
    main()
