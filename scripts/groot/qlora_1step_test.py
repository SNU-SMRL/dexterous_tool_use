"""Gate 2.5: QLoRA 1-step forward+backward test on RTX 5080 16GB.

Reuses GR00T's official training pipeline but replaces the model with a
4-bit quantized + LoRA version. Runs exactly 1 training step to verify
that the full pipeline fits in 16GB VRAM.
"""
import subprocess
import sys

sys.path.insert(0, "Isaac-GR00T")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, BitsAndBytesConfig

from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import MODEL_REGISTRY
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7Config  # noqa: F401
from transformers import TrainingArguments


def get_vram_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip())


if __name__ == "__main__":
    tag = EmbodimentTag.resolve("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": ["Isaac-GR00T/demo_data/droid_sample"],
                        "mix_ratio": 1.0,
                        "embodiment_tag": tag.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = "nvidia/GR00T-N1.7-3B"
    config.training.use_wandb = False
    config.training.num_gpus = 1

    # Setup data pipeline (uses the official pipeline, no model load yet)
    from pathlib import Path

    save_cfg_dir = Path("/tmp/qlora_test_cfg")
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()

    # Get dataset and collator, then discard the full-precision model
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    del pipeline.model
    del pipeline
    torch.cuda.empty_cache()

    # Load model with 4-bit quantization + LoRA
    print("Loading 4-bit quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["action_head", "lm_head"],
    )
    model = AutoModel.from_pretrained("nvidia/GR00T-N1.7-3B", quantization_config=bnb_config)
    model.action_head.to(torch.bfloat16)

    # Collect only quantized (4-bit) backbone linear layer names for LoRA
    import bitsandbytes as bnb

    backbone_linears = []
    for name, module in model.named_modules():
        if (
            name.startswith("backbone.")
            and isinstance(module, bnb.nn.Linear4bit)
            and hasattr(module.weight, "compress_statistics")
        ):
            backbone_linears.append(name)
    print(f"LoRA target: {len(backbone_linears)} backbone Linear layers")

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=backbone_linears)
    model = get_peft_model(model, lora_config)

    # Keep action_head frozen — only LoRA adapters on backbone are trained
    model.gradient_checkpointing_enable()

    # Fix: beta_dist concentration is cast to fp16 under autocast, but
    # PyTorch Dirichlet kernel requires float32. Rebuild in fp32 on CPU.
    from torch.distributions import Beta

    ah = model.base_model.model.action_head
    c1 = ah.beta_dist.concentration1.float().cpu()
    c0 = ah.beta_dist.concentration0.float().cpu()
    ah.beta_dist = Beta(c1, c0)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.1f}%)")
    print(f"VRAM after model load: {get_vram_mb()} MB")

    training_args = TrainingArguments(
        output_dir="/tmp/qlora_test",
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=999999,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ignore_data_skip=True,
    )

    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting 1-step training...")
    trainer.train()

    print(f"VRAM after forward+backward: {get_vram_mb()} MB")
    print("PASS")
