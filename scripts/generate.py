"""
LIVO Text Generation Script
Usage: python scripts/generate.py --checkpoint checkpoints/latest.pt [--prompt "Once upon a time"]
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tokenizer import livorator
from model.llm import LLM, Config
from utils.device import resolve_device, configure_runtime
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with LIVO LLM")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or directory")
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs" / "model.yml"),
    )
    parser.add_argument("--prompt", default="Once upon a time", help="Starting text")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_checkpoint_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        latest = path / "latest.pt"
        if latest.exists():
            return latest
        checkpoints = sorted(path.glob("step_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        raise FileNotFoundError(f"No checkpoints found in: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    logger = get_logger("livo")

    # 1. Setup
    device = resolve_device(args.device)
    configure_runtime(device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)

    # 2. Load config and build model
    with open(args.model_config) as f:
        model_cfg_dict = yaml.safe_load(f) or {}
    config = Config.from_dict(model_cfg_dict)
    model = LLM(config).to(device)

    # 3. Load checkpoint weights
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded from step %d", ckpt.get("step", 0))

    # 4. Setup tokenizer
    tokenizer = livorator(vocab_size=config.vocab_size, max_length=config.max_length)

    # 5. Generate
    logger.info("Prompt: %s", args.prompt)

    text = tokenizer.generate_text(
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=str(device),
    )

    # 6. Output
    print(f"\n{'='*50}")
    print("Generated Text:")
    print(f"{'='*50}")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("unicode_escape").decode("ascii"))
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
