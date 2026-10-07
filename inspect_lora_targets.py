###用于检查LoRA挂的层对不对
import torch
from peft import LoraConfig, TaskType, get_peft_model

from starvector.model.starvector_arch import (
    StarVectorForCausalLM,
)


MODEL_NAME = "starvector/starvector-1b-im2svg"

LANGUAGE_PREFIX = (
    "model.svg_transformer.transformer.transformer.h."
)

TARGET_SUFFIXES = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)

EXPECTED_LAYERS = 24
EXPECTED_TARGETS_PER_LAYER = 4
EXPECTED_TARGET_COUNT = (
    EXPECTED_LAYERS * EXPECTED_TARGETS_PER_LAYER
)


def find_lora_targets(model):
    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if not name.startswith(LANGUAGE_PREFIX):
            continue

        if not name.endswith(TARGET_SUFFIXES):
            continue

        targets.append(name)

    return sorted(targets)


def count_parameters(model):
    total = 0
    trainable = 0

    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total += parameter_count

        if parameter.requires_grad:
            trainable += parameter_count

    return total, trainable


def main():
    print(f"Loading model: {MODEL_NAME}")

    model = StarVectorForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # 先冻结全部原始参数。
    model.requires_grad_(False)

    target_modules = find_lora_targets(model)

    print()
    print("LoRA target modules:")
    print("-" * 80)

    for index, name in enumerate(target_modules, start=1):
        print(f"{index:03d}: {name}")

    print("-" * 80)
    print(f"Found targets: {len(target_modules)}")
    print(f"Expected targets: {EXPECTED_TARGET_COUNT}")

    if len(target_modules) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(
            "LoRA target count mismatch: "
            f"found {len(target_modules)}, "
            f"expected {EXPECTED_TARGET_COUNT}"
        )

    forbidden_targets = [
        name
        for name in target_modules
        if (
            "image_encoder" in name
            or "image_projection" in name
        )
    ]

    if forbidden_targets:
        raise RuntimeError(
            "LoRA targets incorrectly include visual modules:\n"
            + "\n".join(forbidden_targets)
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )

    print()
    print("Injecting LoRA adapters...")

    lora_model = get_peft_model(
        model,
        lora_config,
    )

    total_parameters, trainable_parameters = (
        count_parameters(lora_model)
    )

    trainable_ratio = (
        100 * trainable_parameters / total_parameters
    )

    print()
    print("Parameter summary:")
    print(f"Total parameters: {total_parameters:,}")
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(f"Trainable ratio: {trainable_ratio:.6f}%")

    trainable_names = [
        name
        for name, parameter
        in lora_model.named_parameters()
        if parameter.requires_grad
    ]

    invalid_trainable = [
        name
        for name in trainable_names
        if "lora_" not in name
    ]

    if invalid_trainable:
        raise RuntimeError(
            "Non-LoRA parameters are unexpectedly trainable:\n"
            + "\n".join(invalid_trainable)
        )

    visual_trainable = [
        name
        for name in trainable_names
        if (
            "image_encoder" in name
            or "image_projection" in name
        )
    ]

    if visual_trainable:
        raise RuntimeError(
            "Visual modules are unexpectedly trainable:\n"
            + "\n".join(visual_trainable)
        )

    print()
    print("Trainable LoRA tensors:")
    print("-" * 80)

    for name in trainable_names:
        print(name)

    print("-" * 80)
    print(f"Trainable tensors: {len(trainable_names)}")

    # 每个目标线性层应产生 lora_A 和 lora_B。
    expected_trainable_tensors = (
        EXPECTED_TARGET_COUNT * 2
    )

    if len(trainable_names) != expected_trainable_tensors:
        raise RuntimeError(
            "Unexpected number of trainable LoRA tensors: "
            f"found {len(trainable_names)}, "
            f"expected {expected_trainable_tensors}"
        )

    print()
    print("LoRA target inspection passed.")
    print(
        "Only the 96 StarCoder attention/FFN modules "
        "received LoRA adapters."
    )
    print(
        "Image encoder and image projection remain frozen."
    )


if __name__ == "__main__":
    main()