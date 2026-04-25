from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from training.loss import causal_lm_loss, perplexity_from_loss
from utils.device import autocast_context, move_to_device, use_grad_scaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainerConfig:
    seed: int = 42
    precision: str = "fp16"
    batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 10000
    grad_clip_norm: float = 1.0
    save_dir: str = "checkpoints"
    save_every: int = 100
    log_every: int = 10
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    num_workers: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        optimizer_cfg = data.get("optimizer", {})
        checkpoint_cfg = data.get("checkpoint", {})
        logging_cfg = data.get("logging", {})
        data_cfg = data.get("data", {})

        return cls(
            seed=int(data.get("seed", cls.seed)),
            precision=str(data.get("precision", cls.precision)),
            batch_size=int(data.get("batch_size", cls.batch_size)),
            grad_accum_steps=int(data.get("grad_accum_steps", cls.grad_accum_steps)),
            learning_rate=float(data.get("learning_rate", cls.learning_rate)),
            weight_decay=float(data.get("weight_decay", cls.weight_decay)),
            warmup_steps=int(data.get("warmup_steps", cls.warmup_steps)),
            max_steps=int(data.get("max_steps", cls.max_steps)),
            grad_clip_norm=float(data.get("grad_clip_norm", cls.grad_clip_norm)),
            save_dir=str(checkpoint_cfg.get("save_dir", cls.save_dir)),
            save_every=int(checkpoint_cfg.get("save_every", cls.save_every)),
            log_every=int(logging_cfg.get("log_every", cls.log_every)),
            betas=tuple(optimizer_cfg.get("betas", cls.betas)),
            eps=float(optimizer_cfg.get("eps", cls.eps)),
            num_workers=int(data_cfg.get("num_workers", cls.num_workers)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_optimizer(model: torch.nn.Module, config: TrainerConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if config.max_steps <= 0:
            return 1.0
        if step < config.warmup_steps:
            return float(step + 1) / float(max(1, config.warmup_steps))

        progress = (step - config.warmup_steps) / float(max(1, config.max_steps - config.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class Trainer:
    """Minimal trainer for causal language modeling."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        config: TrainerConfig,
        device: torch.device,
        logger,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        model_config: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.optimizer = optimizer or build_optimizer(self.model, config)
        self.scheduler = scheduler or build_scheduler(self.optimizer, config)
        self.model_config = model_config or {}
        self.train_config = train_config or config.to_dict()
        self.global_step = 0

        scaler_enabled = use_grad_scaler(device, config.precision)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.checkpoint_dir = Path(config.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, step: Optional[int] = None) -> Path:
        step = self.global_step if step is None else step
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "trainer_config": self.config.to_dict(),
            "model_config": self.model_config,
            "train_config": self.train_config,
        }

        checkpoint_path = self.checkpoint_dir / f"step_{step:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_file: str | Path, fine_tune: bool = False) -> Dict[str, Any]:
        """
        Loads a checkpoint. 
        Set fine_tune=True to reset steps/scheduler for a new task.
        Set fine_tune=False to resume exactly where you left off.
        """
        try:
            checkpoint = torch.load(
                checkpoint_file, 
                map_location=self.device, 
                weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)

        # 1. Always load the weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if fine_tune:
            # 🔥 FINE-TUNING MODE: Start fresh but keep the smart weights
            self.global_step = 0
            for g in self.optimizer.param_groups:
                g["lr"] = self.config.learning_rate
            self.scheduler = build_scheduler(self.optimizer, self.config)
            self.logger.info("Loaded weights for FINE-TUNING (Step 0 reset)")
        else:
            # 🔄 RESUME MODE: Continue pre-training exactly where you left off
            self.global_step = checkpoint.get("step", 0)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.logger.info(f"Resuming SCRATCH training from step {self.global_step}")

        return checkpoint

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if outputs.loss is not None:
            return outputs.loss
        return causal_lm_loss(
            outputs.logits,
            batch["labels"],
            ignore_index=self.model.config.pad_token_id,
        )

    def train(self) -> Path:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        accumulated_micro_loss = 0.0
        running_step_loss = 0.0
        optimizer_steps_since_log = 0
        accumulated_tokens = 0
        last_checkpoint = self.checkpoint_dir / "latest.pt"
        t0 = time.time()

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                batch = move_to_device(batch, self.device)
                accumulated_tokens += batch["input_ids"].numel()

                with autocast_context(self.device, self.config.precision):
                    loss = self._forward_loss(batch)
                    scaled_loss = loss / self.config.grad_accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accumulated_micro_loss += float(loss.item())
                micro_step += 1

                if micro_step % self.config.grad_accum_steps != 0:
                    continue

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                optimizer_step_loss = accumulated_micro_loss / self.config.grad_accum_steps
                running_step_loss += optimizer_step_loss
                optimizer_steps_since_log += 1
                accumulated_micro_loss = 0.0

                if self.global_step % self.config.log_every == 0:
                    average_loss = running_step_loss / max(1, optimizer_steps_since_log)
                    learning_rate = self.scheduler.get_last_lr()[0]
                    t1 = time.time()
                    dt = t1 - t0
                    tps = accumulated_tokens / dt if dt > 0 else 0
                    self.logger.info(
                        "step=%d loss=%.4f ppl=%.2f lr=%.6f tps=%.1f",
                        self.global_step,
                        average_loss,
                        perplexity_from_loss(average_loss),
                        learning_rate,
                        tps,
                    )
                    running_step_loss = 0.0
                    optimizer_steps_since_log = 0
                    accumulated_tokens = 0
                    t0 = time.time()

                if self.global_step % self.config.save_every == 0:
                    last_checkpoint = self.save_checkpoint()
                    self.logger.info("Saved checkpoint to %s", last_checkpoint)

                if self.global_step >= self.config.max_steps:
                    break

        if self.global_step % self.config.save_every != 0:
            last_checkpoint = self.save_checkpoint()
            self.logger.info("Saved final checkpoint to %s", last_checkpoint)

        return last_checkpoint