import argparse
import sys
from pathlib import Path

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TextDataset
from data.collator import CausalLMCollator
from model.llm import LLM, Config
from training.trainer import Trainer, TrainerConfig, set_seed
from utils.device import resolve_device, configure_runtime
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LIVO LLM")

    # Config files
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model.yml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "train.yml"))

    # Dataset options
    parser.add_argument("--dataset", default="roneneldan/TinyStories", help="HuggingFace dataset name or local path")
    parser.add_argument("--dataset-config", default=None, help="HuggingFace dataset config name")
    parser.add_argument("--text-column", default="text", help="Text column name in dataset")
    parser.add_argument("--local", action="store_true", help="Treat --dataset as local directory path")

    # Tokenizer
    parser.add_argument("--tokenizer", default=None, help="Path to trained tokenizer JSON")

    # Overrides
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction of data for validation (0 to disable)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("livo")

    # 1. Load configs
    with open(args.model_config) as f:
        model_cfg_dict = yaml.safe_load(f) or {}
    with open(args.train_config) as f:
        train_cfg_dict = yaml.safe_load(f) or {}

    model_config = Config.from_dict(model_cfg_dict)
    train_config = TrainerConfig.from_dict(train_cfg_dict)

    # Apply command-line overrides
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size

    # 2. Setup device
    device = resolve_device(args.device)
    configure_runtime(device)
    logger.info("Device: %s", device)

    # 3. Seed
    set_seed(train_config.seed)
    logger.info("Seed: %d", train_config.seed)

    # 4. Build dataset
    logger.info("Loading dataset: %s", args.dataset)
    dataset = TextDataset(
        source=args.dataset,
        config=args.dataset_config,
        split="train",
        text_column=args.text_column,
        max_length=model_config.max_length,
        vocab_size=model_config.vocab_size,
        tokenizer_path=args.tokenizer,
        max_samples=args.max_samples,
        local=args.local,
    )
    logger.info("Dataset size: %d samples", len(dataset))

    # Split into train/val if requested
    val_loader = None
    if args.val_split > 0 and len(dataset) > 10:
        val_size = max(1, int(len(dataset) * args.val_split))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        logger.info("Split: %d train / %d val (%.0f%%)", train_size, val_size, args.val_split * 100)
    else:
        train_dataset = dataset
        val_dataset = None

    collator = CausalLMCollator(pad_token_id=model_config.pad_token_id)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            collate_fn=collator,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    # 5. Build model
    model = LLM(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s parameters", f"{param_count:,}")

    # 6. Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=train_config,
        device=device,
        logger=logger,
        val_loader=val_loader,
        model_config=model_cfg_dict,
        train_config=train_cfg_dict,
    )

    # 7. Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info("Resumed from: %s", args.resume)

    # 8. Train
    logger.info("Starting training for %d steps...", train_config.max_steps)
    last_checkpoint = trainer.train()
    logger.info("Training complete! Final checkpoint: %s", last_checkpoint)


if __name__ == "__main__":
    main()
