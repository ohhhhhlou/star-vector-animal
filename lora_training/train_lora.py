import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from starvector.data.util import ImageTrainProcessor
from starvector.model.starvector_arch import StarVectorForCausalLM

MODEL_NAME = "starvector/starvector-1b-im2svg"
LANGUAGE_PREFIX = "model.svg_transformer.transformer.transformer.h."
TARGET_SUFFIXES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class AnimalSvgDataset(Dataset):
    def __init__(self, manifest, image_size=224):
        self.manifest = Path(manifest)
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.processor = ImageTrainProcessor(size=image_size)

    def __len__(self):
        return len(self.rows)

    def _resolve(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self._resolve(row["image"])).convert("RGB")
        svg = self._resolve(row["svg"]).read_text(encoding="utf-8")
        return {
            "image": self.processor(image),
            "svg": svg,
            "caption": row.get("prompt", ""),
            "id": row.get("id", f"{index:06d}"),
        }


def collate_batch(rows):
    return {
        "image": torch.stack([row["image"] for row in rows]),
        "svg": [row["svg"] for row in rows],
        "caption": [row["caption"] for row in rows],
        "id": [row["id"] for row in rows],
    }


def move_batch(batch, device):
    batch["image"] = batch["image"].to(device, non_blocking=True)
    return batch


def find_lora_targets(model):
    return sorted(
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith(LANGUAGE_PREFIX)
        and name.endswith(TARGET_SUFFIXES)
    )


def load_model(args):
    base = StarVectorForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    base.requires_grad_(False)
    if args.resume_adapter:
        model = PeftModel.from_pretrained(
            base, args.resume_adapter, is_trainable=True
        )
    else:
        targets = find_lora_targets(base)
        if len(targets) != 96:
            raise RuntimeError(f"Expected 96 LoRA targets, found {len(targets)}")
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
        )
        model = get_peft_model(base, config)
    return model.cuda()


def official_loss(model, batch):
    # The repository's official batch training forward is implemented by
    # StarVectorBase, one level below StarVectorForCausalLM.
    return model.get_base_model().model(batch)


@torch.no_grad()
def validate(model, loader, max_batches):
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = move_batch(batch, "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = official_loss(model, batch)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(model, optimizer, scheduler, scaler, output_dir, step, epoch):
    checkpoint = Path(output_dir) / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint / "trainer_state.pt",
    )
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/animal-lora/train.jsonl")
    parser.add_argument("--validation-manifest", default="data/animal-lora/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/animal-lora")
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_data = AnimalSvgDataset(args.train_manifest)
    validation_data = AnimalSvgDataset(args.validation_manifest)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_batch, generator=generator,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_batch,
    )

    model = load_model(args)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("Expected only LoRA parameters to be trainable")

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    planned_steps = updates_per_epoch * args.epochs
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train/validation samples: {len(train_data)}/{len(validation_data)}")
    print(f"Trainable tensors: {len(trainable)}")
    print(f"Optimizer steps planned: {total_steps}")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = official_loss(model, batch)
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {float(loss)}")
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss)

            should_update = micro_step % args.gradient_accumulation == 0 or micro_step == len(train_loader)
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0:
                average = running_loss / args.gradient_accumulation
                print(f"epoch={epoch + 1} step={global_step} loss={average:.6f} lr={scheduler.get_last_lr()[0]:.3e}", flush=True)
                running_loss = 0.0

            if global_step % args.validate_every == 0:
                value = validate(model, validation_loader, args.validation_batches)
                print(f"validation step={global_step} loss={value:.6f}", flush=True)

            if global_step % args.save_every == 0:
                path = save_checkpoint(model, optimizer, scheduler, scaler, args.output_dir, global_step, epoch)
                print(f"saved={path}", flush=True)

            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    final_dir = Path(args.output_dir) / "final-adapter"
    model.save_pretrained(final_dir, safe_serialization=True)
    final_validation = validate(model, validation_loader, args.validation_batches)
    print(f"final_validation_loss={final_validation:.6f}")
    print(f"final_adapter={final_dir}")
    print("LORA TRAINING COMPLETE")


if __name__ == "__main__":
    main()
