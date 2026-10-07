import gc
import json
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model

from starvector.data.util import ImageTrainProcessor
from starvector.model.starvector_arch import StarVectorForCausalLM


MODEL_NAME = "starvector/starvector-1b-im2svg"
MANIFEST = Path("data/strength-animal/train-strength.jsonl")
OUTPUT_DIR = Path("outputs/lora-strength-adapter")
LANGUAGE_PREFIX = "model.svg_transformer.transformer.transformer.h."
TARGET_SUFFIXES = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def find_lora_targets(model):
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith(LANGUAGE_PREFIX)
        and name.endswith(TARGET_SUFFIXES)
    )


def load_one_batch():
    with MANIFEST.open("r", encoding="utf-8") as handle:
        sample = json.loads(handle.readline())

    image_path = resolve_path(sample["image"])
    svg_path = resolve_path(sample["svg"])
    image = Image.open(image_path).convert("RGB")
    image_tensor = ImageTrainProcessor(size=224)(image).unsqueeze(0)
    svg = svg_path.read_text(encoding="utf-8")

    batch = {
        "image": image_tensor.cuda(non_blocking=True),
        "svg": [svg],
        "id": [sample.get("id", "probe-000000")],
        "caption": [sample.get("prompt", "")],
    }
    print(f"Sample image: {image_path}")
    print(f"Sample SVG: {svg_path}")
    print(f"Image batch shape: {tuple(batch['image'].shape)}")
    print(f"SVG characters: {len(svg)}")
    return batch


def load_base_model():
    return StarVectorForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch = load_one_batch()
    base_model = load_base_model()
    base_model.requires_grad_(False)
    targets = find_lora_targets(base_model)
    if len(targets) != 96:
        raise RuntimeError(f"Expected 96 LoRA targets, found {len(targets)}")

    # Keep task_type unset: generic PeftModel preserves StarVector's custom
    # model(batch) forward contract instead of imposing a text-only signature.
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(base_model, config).cuda()
    model.train()

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if len(trainable) != 192:
        raise RuntimeError(f"Expected 192 trainable tensors, found {len(trainable)}")
    if any("lora_" not in name for name in trainable):
        raise RuntimeError("A non-LoRA parameter is trainable")

    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainable.items()
    }
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model.get_base_model().model(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")
    print(f"Forward loss: {loss.item():.6f}")

    loss.backward()
    grad_present = sum(parameter.grad is not None for parameter in trainable.values())
    grad_nonzero = sum(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in trainable.values()
    )
    if grad_present != 192:
        raise RuntimeError(
            f"Only {grad_present}/192 LoRA tensors received gradients"
        )
    if grad_nonzero == 0:
        raise RuntimeError("All LoRA gradients are zero")

    torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
    optimizer.step()
    changed = sum(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in trainable.items()
    )
    if changed == 0:
        raise RuntimeError("Optimizer step did not update any LoRA tensor")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    expected_files = [OUTPUT_DIR / "adapter_config.json", OUTPUT_DIR / "adapter_model.safetensors"]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing saved adapter files: {missing}")

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Gradient tensors present: {grad_present}/192")
    print(f"Gradient tensors nonzero: {grad_nonzero}/192")
    print(f"LoRA tensors changed after step: {changed}/192")
    print(f"Peak allocated GPU memory: {peak_gib:.3f} GiB")
    print(f"Saved adapter: {OUTPUT_DIR}")

    del optimizer, model, base_model, batch, trainable, before
    gc.collect()
    torch.cuda.empty_cache()

    reload_base = load_base_model()
    reloaded = PeftModel.from_pretrained(
        reload_base,
        OUTPUT_DIR,
        is_trainable=False,
    )
    adapter_tensor_count = sum(
        "lora_" in name for name, _ in reloaded.named_parameters()
    )
    if adapter_tensor_count != 192:
        raise RuntimeError(
            f"Reloaded adapter tensor count mismatch: {adapter_tensor_count}"
        )

    print(f"Reloaded adapter tensors: {adapter_tensor_count}/192")
    print("LORA ONE-BATCH PROBE PASSED")


if __name__ == "__main__":
    main()
