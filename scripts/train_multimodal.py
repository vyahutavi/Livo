"""
LIVO Multimodal Training Script — Train vision, audio, and speech components.

Usage:
    # Text-only (same as train.py)
    python scripts/train_multimodal.py --mode text

    # Vision + Text (requires image-text dataset)
    python scripts/train_multimodal.py --mode vision-text --dataset <dataset>

    # Full multimodal (all modalities)
    python scripts/train_multimodal.py --mode full --dataset <dataset>

    # Freeze LLM, train only encoders
    python scripts/train_multimodal.py --mode vision-text --freeze-llm

    # Resume from checkpoint
    python scripts/train_multimodal.py --mode full --resume checkpoints/latest.pt
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.llm import Config
from model.multimodal import MultimodalLIVO, MultimodalConfig
from model.audio_encoder import AudioEncoderConfig
from model.vision_encoder import VisionEncoderConfig
from model.speech_decoder import SpeechDecoderConfig
from training.trainer import Trainer, TrainerConfig, set_seed
from utils.device import resolve_device, configure_runtime
from utils.logger import get_logger


MODES = {
    "text": "Text-only (LLM brain only)",
    "vision-text": "Vision encoder + LLM",
    "audio-text": "Audio encoder + LLM",
    "full": "All modalities (vision + audio + text + speech)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LIVO Multimodal Model")

    # Mode
    parser.add_argument(
        "--mode", default="text", choices=MODES.keys(),
        help="Training mode: " + ", ".join(f"{k} ({v})" for k, v in MODES.items()),
    )

    # Config files
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model.yml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "train.yml"))
    parser.add_argument("--vision-config", default=str(PROJECT_ROOT / "configs" / "vision.yml"))
    parser.add_argument("--audio-config", default=str(PROJECT_ROOT / "configs" / "audio.yml"))
    parser.add_argument("--speech-config", default=str(PROJECT_ROOT / "configs" / "speech.yml"))

    # Dataset (placeholder — user will specify later)
    parser.add_argument("--dataset", default=None, help="Dataset name or path")
    parser.add_argument("--dataset-config", default=None, help="HuggingFace dataset config")

    # Training strategy
    parser.add_argument("--freeze-llm", action="store_true", help="Freeze LLM weights, train only encoders")
    parser.add_argument("--freeze-encoders", action="store_true", help="Freeze encoders, train only LLM")
    parser.add_argument("--llm-checkpoint", default=None, help="Pre-trained LLM checkpoint to load")

    # Overrides
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_multimodal_config(args) -> MultimodalConfig:
    """Build MultimodalConfig based on the selected training mode."""
    model_cfg = Config.from_dict(load_yaml(args.model_config))

    vision_cfg = None
    audio_cfg = None
    speech_cfg = None

    if args.mode in ("vision-text", "full"):
        raw = load_yaml(args.vision_config)
        vision_data = raw.get("vision_encoder", raw)
        vision_data.setdefault("d_model", model_cfg.d_model)
        vision_cfg = VisionEncoderConfig(**{
            k: v for k, v in vision_data.items()
            if k in VisionEncoderConfig.__dataclass_fields__
        })

    if args.mode in ("audio-text", "full"):
        raw = load_yaml(args.audio_config)
        audio_data = raw.get("audio_encoder", raw)
        audio_data.setdefault("d_model", model_cfg.d_model)
        audio_cfg = AudioEncoderConfig(**{
            k: v for k, v in audio_data.items()
            if k in AudioEncoderConfig.__dataclass_fields__
        })

    if args.mode == "full":
        raw = load_yaml(args.speech_config)
        speech_data = raw.get("speech_decoder", raw)
        speech_data.setdefault("d_model", model_cfg.d_model)
        speech_cfg = SpeechDecoderConfig(**{
            k: v for k, v in speech_data.items()
            if k in SpeechDecoderConfig.__dataclass_fields__
        })

    return MultimodalConfig(
        llm=model_cfg,
        vision=vision_cfg,
        audio=audio_cfg,
        speech=speech_cfg,
    )


def main() -> None:
    args = parse_args()
    logger = get_logger("livo")

    # 1. Setup
    device = resolve_device(args.device)
    configure_runtime(device)
    logger.info("Mode: %s — %s", args.mode, MODES[args.mode])
    logger.info("Device: %s", device)

    # 2. Load configs
    train_cfg_dict = load_yaml(args.train_config)
    train_config = TrainerConfig.from_dict(train_cfg_dict)
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size

    set_seed(train_config.seed)

    # 3. Build model
    mm_config = build_multimodal_config(args)
    model = MultimodalLIVO(mm_config)

    # Load pre-trained LLM weights if provided
    if args.llm_checkpoint:
        logger.info("Loading pre-trained LLM from: %s", args.llm_checkpoint)
        try:
            ckpt = torch.load(args.llm_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.llm_checkpoint, map_location="cpu")
        model.llm.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded LLM weights from step %d", ckpt.get("step", 0))

    # Apply freeze strategy
    if args.freeze_llm:
        model.freeze_llm()
        logger.info("Froze LLM weights (training encoders only)")
    if args.freeze_encoders:
        model.freeze_encoders()
        logger.info("Froze encoder weights (training LLM only)")

    # Parameter report
    params = model.num_parameters
    logger.info("Parameter breakdown:")
    for component, count in params.items():
        logger.info("  %s: %s", component, f"{count:,}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.1f%%)", f"{trainable:,}", f"{total:,}", trainable / total * 100)

    # 4. Dataset placeholder
    if args.dataset is None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  MODEL BUILT SUCCESSFULLY — No dataset specified.")
        logger.info("")
        logger.info("  To train, provide a dataset:")
        logger.info("    --dataset <huggingface_dataset_name>")
        logger.info("    --dataset <local_path> (for local files)")
        logger.info("")
        logger.info("  Multimodal datasets need custom DataLoader logic")
        logger.info("  based on your specific data format.")
        logger.info("=" * 60)

        # Quick forward pass test
        logger.info("")
        logger.info("Running quick forward pass test...")
        model = model.to(device)
        text_ids = torch.randint(0, mm_config.llm.vocab_size, (1, 16)).to(device)

        test_kwargs = {"text_ids": text_ids, "labels": text_ids}
        if mm_config.vision is not None:
            test_kwargs["image"] = torch.randn(1, 3, 224, 224).to(device)
        if mm_config.audio is not None:
            test_kwargs["audio"] = torch.randn(1, 80, 100).to(device)

        with torch.no_grad():
            out = model(**test_kwargs)

        logger.info("  Logits shape: %s", tuple(out.logits.shape))
        logger.info("  Loss: %.4f", out.loss.item())
        logger.info("  ✅ Forward pass OK — Model is ready for training!")
        return

    # 5. If dataset IS provided, we would build DataLoaders here.
    #    This is a placeholder for future multimodal dataset integration.
    logger.info("Dataset: %s", args.dataset)
    logger.info("Multimodal dataset loading will be implemented based on your chosen dataset format.")
    logger.info("For text-only training, use: python scripts/train.py --dataset %s", args.dataset)


if __name__ == "__main__":
    main()
