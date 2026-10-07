# Hugging Face evaluation backend for StarVector.
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset

from starvector.data.util import ImageTrainProcessor, rasterize_svg
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.validation.svg_validator_base import SVGValidator, register_validator


class SVGValDataset(Dataset):
    def __init__(self, dataset_name, config_name, split, im_size, num_samples, processor):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.im_size = im_size
        self.num_samples = num_samples
        self.processor = processor

        if self.config_name:
            self.data = load_dataset(self.dataset_name, self.config_name, split=self.split)
        else:
            self.data = load_dataset(self.dataset_name, split=self.split)

        if self.num_samples != -1:
            self.data = self.data.select(range(min(self.num_samples, len(self.data))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        svg_str = self.data[idx]["Svg"]
        sample_id = self.data[idx]["Filename"]
        image = rasterize_svg(svg_str, resolution=self.im_size)
        image = self.processor(image)
        caption = self.data[idx].get("Caption", "")
        return {
            "Svg": svg_str,
            "image": image,
            "Filename": sample_id,
            "Caption": caption,
        }


@register_validator
class StarVectorHFSVGValidator(SVGValidator):
    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.model.torch_dtype]

        model_path = self.resume_from_checkpoint if config.model.from_checkpoint else config.model.name
        self.model = StarVectorForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=config.model.get("local_files_only", False),
        ).to(config.run.device)
        self.model.eval()

        self.processor = ImageTrainProcessor(size=config.dataset.im_size)
        self.tokenizer = self.model.model.svg_transformer.tokenizer
        self.svg_end_token_id = self.tokenizer.encode("</svg>")[0]
        self.get_dataloader()

    def get_dataloader(self):
        self.dataset = SVGValDataset(
            self.config.dataset.dataset_name,
            self.config.dataset.config_name,
            self.config.dataset.split,
            self.config.dataset.im_size,
            self.config.dataset.num_samples,
            self.processor,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
        )
        return self.dataloader

    def release_memory(self):
        self.model.model.svg_transformer.tokenizer = None
        self.model.model.svg_transformer.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def generate_svg(self, batch, generate_config):
        generate_config = dict(generate_config)
        if generate_config.get("temperature", 1.0) == 0:
            generate_config["temperature"] = 1.0
            generate_config["do_sample"] = False

        batch["image"] = batch["image"].to(
            device=self.config.run.device,
            dtype=self.torch_dtype,
        )
        if self.task == "im2svg":
            return self.model.generate_im2svg(batch=batch, **generate_config)
        if self.task == "text2svg":
            return self.model.generate_text2svg(batch=batch, **generate_config)
        raise ValueError(f"Unsupported task: {self.task}")
