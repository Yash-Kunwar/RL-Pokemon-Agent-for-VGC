import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from typing import List, Optional
from collections import defaultdict

from src.data.replay_parser import load_battles_from_file, ParsedBattle
from src.data.dataset import VGCBattleDataset
from src.models.transformer_policy import VGCPolicyNetwork
from src.utils.action_space import N_ACTIONS, PASS_ACTION

# ─── Training Config ───────────────────────────────────────────────────────────

class BCConfig:
    # Data
    DATA_FILES = [
        'data/replays/logs_gen9vgc2025regi.json',
        'data/replays/logs_gen9vgc2025regh.json',
        'data/replays/logs_gen9vgc2024regh.json',
        'data/replays/logs_gen9vgc2024regg.json',
    ]
    MAX_BATTLES_PER_FILE = None     # None = load all
    VAL_SPLIT = 0.1                 # 10% validation
    SKIP_PASS_ONLY = True

    # Training
    BATCH_SIZE = 256
    NUM_EPOCHS = 15
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0                 # gradient clipping max norm

    # Label smoothing — prevents overconfidence
    LABEL_SMOOTHING = 0.1

    # Pass action weight — downweight pass since it's trivial
    PASS_ACTION_WEIGHT = 0.1

    # Scheduler — cosine annealing
    USE_SCHEDULER = True

    # Checkpointing
    CHECKPOINT_DIR = 'logs/checkpoints'
    SAVE_EVERY_N_EPOCHS = 1

    # Hardware
    NUM_WORKERS = 0                 # set to 4 if on Linux
    PIN_MEMORY = False              # set True if using CUDA


# ─── Loss Function ─────────────────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropy loss with:
    - Label smoothing to prevent overconfidence
    - Per-class weights to downweight trivial actions (pass)
    """
    def __init__(self, n_classes: int, pass_action: int, pass_weight: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        weights = torch.ones(n_classes)
        weights[pass_action] = pass_weight
        self.register_buffer('weights', weights)
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            labels,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
        )


# ─── Metrics ──────────────────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_loss_0 = 0.0
        self.total_loss_1 = 0.0
        self.correct_0 = 0
        self.correct_1 = 0
        self.correct_both = 0
        self.total = 0
        self.action_counts = defaultdict(int)
        self.action_correct = defaultdict(int)

    def update(
        self,
        loss: float,
        loss_0: float,
        loss_1: float,
        logits_0: torch.Tensor,
        logits_1: torch.Tensor,
        labels_0: torch.Tensor,
        labels_1: torch.Tensor,
    ):
        batch_size = labels_0.size(0)
        self.total += batch_size
        self.total_loss += loss * batch_size
        self.total_loss_0 += loss_0 * batch_size
        self.total_loss_1 += loss_1 * batch_size

        pred_0 = logits_0.argmax(dim=-1)
        pred_1 = logits_1.argmax(dim=-1)

        self.correct_0 += (pred_0 == labels_0).sum().item()
        self.correct_1 += (pred_1 == labels_1).sum().item()
        self.correct_both += ((pred_0 == labels_0) & (pred_1 == labels_1)).sum().item()

        # Per-action accuracy tracking
        for label, pred in zip(labels_0.tolist(), pred_0.tolist()):
            self.action_counts[label] += 1
            if label == pred:
                self.action_correct[label] += 1

    def summary(self) -> dict:
        if self.total == 0:
            return {}
        return {
            'loss':      self.total_loss / self.total,
            'loss_0':    self.total_loss_0 / self.total,
            'loss_1':    self.total_loss_1 / self.total,
            'acc_0':     self.correct_0 / self.total,
            'acc_1':     self.correct_1 / self.total,
            'acc_both':  self.correct_both / self.total,
        }


# ─── Trainer ──────────────────────────────────────────────────────────────────

class BCTrainer:
    def __init__(self, config: BCConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create checkpoint dir
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # Build model
        self.model = VGCPolicyNetwork().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

        # Loss function
        self.criterion = WeightedCrossEntropyLoss(
            n_classes=N_ACTIONS,
            pass_action=PASS_ACTION,
            pass_weight=config.PASS_ACTION_WEIGHT,
            label_smoothing=config.LABEL_SMOOTHING,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_val_acc = 0.0
        self.history = []

    def load_data(self):
        """Load all replay files and build train/val datasets."""
        print("\n=== Loading Data ===")
        all_battles: List[ParsedBattle] = []

        for filepath in self.config.DATA_FILES:
            battles = load_battles_from_file(
                filepath,
                max_battles=self.config.MAX_BATTLES_PER_FILE,
            )
            all_battles.extend(battles)

        print(f"\nTotal battles loaded: {len(all_battles):,}")

        # Shuffle battles before split
        import random
        random.shuffle(all_battles)

        # Train/val split at battle level (not sample level)
        n_val = int(len(all_battles) * self.config.VAL_SPLIT)
        n_train = len(all_battles) - n_val
        train_battles = all_battles[:n_train]
        val_battles = all_battles[n_train:]

        print(f"Train battles: {len(train_battles):,}")
        print(f"Val battles:   {len(val_battles):,}")

        print("\nBuilding train dataset...")
        train_dataset = VGCBattleDataset(
            train_battles,
            skip_pass_only=self.config.SKIP_PASS_ONLY,
        )

        print("\nBuilding val dataset...")
        val_dataset = VGCBattleDataset(
            val_battles,
            skip_pass_only=self.config.SKIP_PASS_ONLY,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
        )

        print(f"\nTrain samples: {len(train_dataset):,}")
        print(f"Val samples:   {len(val_dataset):,}")
        print(f"Train batches: {len(self.train_loader):,}")
        print(f"Val batches:   {len(self.val_loader):,}")

        # Setup scheduler now that we know steps per epoch
        if self.config.USE_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS * len(self.train_loader),
                eta_min=1e-6,
            )

    def _train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        metrics = MetricsTracker()
        start_time = time.time()

        for batch_idx, (obs, labels_0, labels_1) in enumerate(self.train_loader):
            obs = obs.to(self.device)
            labels_0 = labels_0.to(self.device)
            labels_1 = labels_1.to(self.device)

            # Forward pass — no action masks during BC training
            logits_0, logits_1, _ = self.model(obs)

            # Compute loss for both slots
            loss_0 = self.criterion(logits_0, labels_0)
            loss_1 = self.criterion(logits_1, labels_1)
            loss = (loss_0 + loss_1) / 2.0

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRAD_CLIP,
            )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            with torch.no_grad():
                metrics.update(
                    loss.item(), loss_0.item(), loss_1.item(),
                    logits_0, logits_1, labels_0, labels_1,
                )

            # Progress logging
            if (batch_idx + 1) % 100 == 0:
                summary = metrics.summary()
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={summary['loss']:.4f} "
                    f"acc0={summary['acc_0']:.3f} "
                    f"acc1={summary['acc_1']:.3f} "
                    f"lr={lr:.2e} "
                    f"({elapsed:.0f}s)"
                )

        return metrics.summary()

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        """Run one validation epoch."""
        self.model.eval()
        metrics = MetricsTracker()

        for obs, labels_0, labels_1 in self.val_loader:
            obs = obs.to(self.device)
            labels_0 = labels_0.to(self.device)
            labels_1 = labels_1.to(self.device)

            logits_0, logits_1, _ = self.model(obs)

            loss_0 = self.criterion(logits_0, labels_0)
            loss_1 = self.criterion(logits_1, labels_1)
            loss = (loss_0 + loss_1) / 2.0

            metrics.update(
                loss.item(), loss_0.item(), loss_1.item(),
                logits_0, logits_1, labels_0, labels_1,
            )

        return metrics.summary()

    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config.__dict__,
        }

        # Save latest
        path = os.path.join(self.config.CHECKPOINT_DIR, f'bc_epoch_{epoch:02d}.pt')
        torch.save(checkpoint, path)

        # Save best separately
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'bc_best.pt')
            torch.save(checkpoint, best_path)
            print(f"  *** New best model saved (val_acc_both={val_metrics['acc_both']:.4f}) ***")

    def train(self):
        """Full training loop."""
        if self.train_loader is None:
            self.load_data()

        print(f"\n=== Starting Behavioral Cloning Training ===")
        print(f"Epochs:     {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"LR:         {self.config.LEARNING_RATE}")
        print()

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{self.config.NUM_EPOCHS} ---")
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._val_epoch()

            epoch_time = time.time() - epoch_start

            # Check if best
            val_acc_both = val_metrics.get('acc_both', 0)
            is_best = val_acc_both > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc_both

            # Save checkpoint
            if epoch % self.config.SAVE_EVERY_N_EPOCHS == 0:
                self._save_checkpoint(epoch, val_metrics, is_best)

            # Log epoch summary
            print(
                f"\nEpoch {epoch} Summary ({epoch_time:.0f}s):\n"
                f"  Train: loss={train_metrics['loss']:.4f} "
                f"acc0={train_metrics['acc_0']:.3f} "
                f"acc1={train_metrics['acc_1']:.3f} "
                f"acc_both={train_metrics['acc_both']:.3f}\n"
                f"  Val:   loss={val_metrics['loss']:.4f} "
                f"acc0={val_metrics['acc_0']:.3f} "
                f"acc1={val_metrics['acc_1']:.3f} "
                f"acc_both={val_metrics['acc_both']:.3f}"
            )

            self.history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
            })

        print(f"\n=== Training Complete ===")
        print(f"Best val acc_both: {self.best_val_acc:.4f}")
        return self.history


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create __init__.py for training module
    os.makedirs('src/training', exist_ok=True)
    init_path = 'src/training/__init__.py'
    if not os.path.exists(init_path):
        open(init_path, 'w').close()

    config = BCConfig()
    trainer = BCTrainer(config)
    trainer.load_data()
    trainer.train()