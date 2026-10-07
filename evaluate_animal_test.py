import argparse
import gc
import json
import random
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
        description="Evaluate either the base model or one LoRA adapter on animal test.jsonl."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("base", "lora"),
        help="Load exactly one variant for this run.",
    )
    parser.add_argument(
        "--manifest", default="data/animal-lora/test.jsonl"
    )
    parser.add_argument(
        "--model", default="starvector/starvector-1b-im2svg"
    )
    parser.add_argument(
        "--adapter", default="outputs/animal-lora/final-adapter"
    )
    parser.add_argument(
        "--output-root", default="outputs/animal-eval"
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only this many samples; useful for a smoke test.",
    )
    parser.add_argument(
        "--skip-perceptual-metrics",
        action="store_true",
        help="Skip LPIPS and DinoScore.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate samples even if generated-raw.txt already exists.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {error}"
                    ) from error
    return records


def resolve_path(value, manifest_path):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Referenced file not found: {value}")


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


def json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_safe),
        encoding="utf-8",
    )


def load_selected_model(args):
    print(f"Loading base checkpoint: {args.model}", flush=True)
    base = StarVectorForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    tokenizer = base.model.svg_transformer.tokenizer

    if args.variant == "lora":
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: {adapter_path.resolve()}"
            )
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            is_trainable=False,
        )
    else:
        model = base

    model = model.cuda().eval()
    return model, base, tokenizer


def prepare_sample(record, position, manifest_path, output_root):
    required = {"image", "svg"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Sample {position} missing fields: {sorted(missing)}")

    sample_id = str(record.get("id", f"{position:06d}"))
    sample_dir = output_root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_path = resolve_path(record["image"], manifest_path)
    svg_path = resolve_path(record["svg"], manifest_path)
    return sample_id, sample_dir, image_path, svg_path


def generate_samples(args, manifest_path, records, output_root):
    model, base, tokenizer = load_selected_model(args)
    generated_records = []

    try:
        for position, record in enumerate(records):
            absolute_index = args.start_index + position
            sample_seed = absolute_index
            set_seed(sample_seed)
            sample_id, sample_dir, image_path, svg_path = prepare_sample(
                record, absolute_index, manifest_path, output_root
            )
            raw_path = sample_dir / "generated-raw.txt"
            result_path = sample_dir / "result.json"

            print(
                f"[generate {position + 1}/{len(records)}] "
                f"variant={args.variant} id={sample_id}",
                flush=True,
            )

            ground_truth_svg = svg_path.read_text(encoding="utf-8")
            ground_truth_copy = sample_dir / "ground-truth.svg"
            if not ground_truth_copy.exists() or args.overwrite:
                shutil.copyfile(svg_path, ground_truth_copy)

            if raw_path.exists() and not args.overwrite:
                generated_raw = raw_path.read_text(encoding="utf-8")
                inference_seconds = None
                peak_allocated_gib = None
                peak_reserved_gib = None
                print("  reusing existing generated-raw.txt", flush=True)
            else:
                input_image = Image.open(image_path).convert("RGB")
                processed_image = base.process_images([input_image])[0].cuda()
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
                peak_allocated_gib = (
                    torch.cuda.max_memory_allocated() / 1024**3
                )
                peak_reserved_gib = (
                    torch.cuda.max_memory_reserved() / 1024**3
                )
                raw_path.write_text(generated_raw, encoding="utf-8")
                del processed_image

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
            generated_svg_path = sample_dir / "generated.svg"
            generated_svg_path.write_text(generated_svg, encoding="utf-8")

            generated_tokens = len(
                tokenizer.encode(generated_raw, add_special_tokens=False)
            )
            ground_truth_tokens = len(
                tokenizer.encode(ground_truth_svg, add_special_tokens=False)
            )
            has_svg_start_tag = "<svg" in generated_raw
            has_svg_end_tag = "</svg>" in generated_raw
            likely_hit_max_length = (
                generated_tokens
                >= OFFICIAL_GENERATION_PARAMS["max_length"] - 300
                and not has_svg_end_tag
            )

            sample_result = {
                "variant": args.variant,
                "position": absolute_index,
                "id": sample_id,
                "dataset_index": record.get("dataset_index"),
                "image": str(image_path),
                "ground_truth_svg": str(svg_path),
                "prompt": record.get("prompt"),
                "seed": sample_seed,
                "generation_status": {
                    "has_svg_start_tag": has_svg_start_tag,
                    "has_svg_end_tag": has_svg_end_tag,
                    "likely_hit_max_length": likely_hit_max_length,
                },
                "validity": {
                    "raw_xml_valid": raw_xml_valid,
                    "raw_xml_error": raw_xml_error,
                    "raw_official_parser_valid": raw_parser_valid,
                    "raw_official_parser_error": raw_parser_error,
                    "clean_svg_error": clean_svg_error,
                    "clean_xml_valid": clean_xml_valid,
                    "clean_xml_error": clean_xml_error,
                    "clean_official_parser_valid": clean_parser_valid,
                    "clean_official_parser_error": clean_parser_error,
                },
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
                "metrics": None,
                "error": None,
            }
            write_json(result_path, sample_result)
            generated_records.append((sample_result, sample_dir))
    finally:
        del model
        del base
        gc.collect()
        torch.cuda.empty_cache()

    return generated_records


def compute_metrics(args, generated_records):
    from starvector.metrics.compute_l2 import L2DistanceCalculator
    from starvector.metrics.compute_SSIM import SSIMDistanceCalculator

    l2_metric = L2DistanceCalculator()
    ssim_metric = SSIMDistanceCalculator()
    lpips_metric = None
    dino_metric = None

    if not args.skip_perceptual_metrics:
        print("Loading LPIPS and DINO metrics after releasing StarVector...", flush=True)
        from starvector.metrics.compute_LPIPS import LPIPSDistanceCalculator
        from starvector.metrics.compute_dino_score import DINOScoreCalculator

        lpips_metric = LPIPSDistanceCalculator(device="cuda")
        dino_metric = DINOScoreCalculator(device="cuda")

    for position, (result, sample_dir) in enumerate(generated_records):
        print(
            f"[metrics {position + 1}/{len(generated_records)}] id={result['id']}",
            flush=True,
        )
        try:
            if not result["validity"]["clean_xml_valid"]:
                raise ValueError("Cleaned generated SVG is invalid")

            ground_truth_svg = Path(
                result["ground_truth_svg"]
            ).read_text(encoding="utf-8")
            generated_svg = (sample_dir / "generated.svg").read_text(
                encoding="utf-8"
            )
            ground_truth_512 = rasterize_svg(
                ground_truth_svg, resolution=512
            ).convert("RGB")
            generated_512 = rasterize_svg(
                generated_svg, resolution=512
            ).convert("RGB")
            ground_truth_512.save(sample_dir / "ground-truth-512.png")
            generated_512.save(sample_dir / "generated-512.png")

            metric_batch = {
                "gt_im": [ground_truth_512],
                "gen_im": [generated_512],
            }
            metrics = {
                "mse_l2_official": unpack_metric(
                    l2_metric.calculate_score(metric_batch, update=False)
                ),
                "ssim_official": unpack_metric(
                    ssim_metric.calculate_score(metric_batch, update=False)
                ),
                "lpips_official": None,
                "dino_score_official": None,
            }
            if lpips_metric is not None:
                metrics["lpips_official"] = unpack_metric(
                    lpips_metric.calculate_score(
                        metric_batch, batch_size=1, update=False
                    )
                )
            if dino_metric is not None:
                metrics["dino_score_official"] = unpack_metric(
                    dino_metric.calculate_score(metric_batch, update=False)
                )
            result["metrics"] = metrics
        except Exception as error:
            result["error"] = f"metric_failure: {error}"
            print(f"  metric failure: {error}", flush=True)

        write_json(sample_dir / "result.json", result)

    del lpips_metric
    del dino_metric
    gc.collect()
    torch.cuda.empty_cache()


def safe_mean(values):
    filtered = [float(value) for value in values if value is not None]
    return float(np.mean(filtered)) if filtered else None


def build_summary(args, manifest_path, total_manifest_size, generated_records):
    results = [item[0] for item in generated_records]
    completed_metrics = [item for item in results if item["metrics"]]

    def metric_values(name):
        return [item["metrics"].get(name) for item in completed_metrics]

    return {
        "dataset": "yoavf/svg-animal-illustrations",
        "split": "test",
        "manifest": str(manifest_path),
        "manifest_size": total_manifest_size,
        "evaluated_samples": len(results),
        "variant": args.variant,
        "base_model": args.model,
        "adapter": args.adapter if args.variant == "lora" else None,
        "protocol": {
            "candidate_count": 1,
            **OFFICIAL_GENERATION_PARAMS,
            "model_input_resolution": 224,
            "metric_resolution": 512,
            "per_sample_seed": "manifest position",
        },
        "counts": {
            "raw_xml_valid": sum(
                item["validity"]["raw_xml_valid"] for item in results
            ),
            "clean_xml_valid": sum(
                item["validity"]["clean_xml_valid"] for item in results
            ),
            "clean_official_parser_valid": sum(
                item["validity"]["clean_official_parser_valid"]
                for item in results
            ),
            "likely_hit_max_length": sum(
                item["generation_status"]["likely_hit_max_length"]
                for item in results
            ),
            "metrics_completed": len(completed_metrics),
            "failures": sum(item["error"] is not None for item in results),
        },
        "rates": {
            "raw_xml_valid_rate": safe_mean(
                [item["validity"]["raw_xml_valid"] for item in results]
            ),
            "clean_xml_valid_rate": safe_mean(
                [item["validity"]["clean_xml_valid"] for item in results]
            ),
        },
        "means": {
            "generated_svg_tokens": safe_mean(
                [item["generated_svg_tokens"] for item in results]
            ),
            "token_length_ratio": safe_mean(
                [item["token_length_ratio"] for item in results]
            ),
            "inference_seconds": safe_mean(
                [item["inference_seconds"] for item in results]
            ),
            "peak_gpu_allocated_gib": safe_mean(
                [item["peak_gpu_allocated_gib"] for item in results]
            ),
            "peak_gpu_reserved_gib": safe_mean(
                [item["peak_gpu_reserved_gib"] for item in results]
            ),
            "mse_l2_official": safe_mean(metric_values("mse_l2_official")),
            "ssim_official": safe_mean(metric_values("ssim_official")),
            "lpips_official": safe_mean(metric_values("lpips_official")),
            "dino_score_official": safe_mean(
                metric_values("dino_score_official")
            ),
        },
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Test manifest not found: {manifest_path.resolve()}"
        )
    all_records = read_jsonl(manifest_path)
    if args.start_index < 0 or args.start_index >= len(all_records):
        raise IndexError(
            f"start-index={args.start_index} outside 0..{len(all_records) - 1}"
        )

    stop = None
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        stop = args.start_index + args.limit
    records = all_records[args.start_index:stop]

    output_root = Path(args.output_root) / args.variant
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "variant": args.variant,
        "manifest": str(manifest_path),
        "base_model": args.model,
        "adapter": args.adapter if args.variant == "lora" else None,
        "start_index": args.start_index,
        "limit": args.limit,
        "skip_perceptual_metrics": args.skip_perceptual_metrics,
        "official_generation_params": OFFICIAL_GENERATION_PARAMS,
    }
    write_json(output_root / "run-config.json", run_config)

    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)
    generated_records = generate_samples(
        args, manifest_path, records, output_root
    )
    compute_metrics(args, generated_records)

    results_jsonl = output_root / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for result, _ in generated_records:
            handle.write(
                json.dumps(result, ensure_ascii=False, default=json_safe)
                + "\n"
            )

    summary = build_summary(
        args, manifest_path, len(all_records), generated_records
    )
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"EVALUATION COMPLETE: {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
