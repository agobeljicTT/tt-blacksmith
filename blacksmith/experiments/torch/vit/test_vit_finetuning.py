# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path

import torch
import torch_xla
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.vit.configs import TrainingConfig
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import collect_examples, show_examples


def validate(model, val_data_loader, loss_fn, device_manager, config, logger):
    logger.info(f"\n=== Starting Validation ===")
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            batch = device_manager.prepare_batch(batch)

            # Forward pass + loss
            inputs = batch["image"]
            labels = batch["label"]
            outputs = model(inputs)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_val_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total_samples += inputs.size(0)

            if config.use_tt:
                torch_xla.sync(wait=True)

            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    accuracy = accuracy * 100
    logger.info(f"Average validation loss: {avg_val_loss}, Accuracy: {accuracy:.2f}%")
    return avg_val_loss, accuracy


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)

    model = model.to(device_manager.device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params}, Trainable%: {trainable_params / total_params * 100:.2f}%")

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    # Load dataset
    train_dataset = get_dataset(config=config, split="train", collate_fn=None)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="test", collate_fn=None)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    # Init training components (optimizer, lr scheduler, etc.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = eval(config.loss_fn)(ignore_index=config.ignored_index)

    global_step = 0
    running_loss = 0.0
    model.train()
    try:
        for epoch in range(config.num_epochs):
            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                batch = device_manager.prepare_batch(batch)

                # Forward pass
                outputs = model(batch["image"])
                logits = outputs.logits

                loss = loss_fn(
                    logits,
                    batch["label"],
                )
                running_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                correct = (predictions == batch["label"]).sum().item()
                total_samples = batch["image"].size(0)
                accuracy = correct / total_samples if total_samples > 0 else 0.0
                accuracy = accuracy * 100

                # Backward pass
                loss.backward()
                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Update parameters
                device_manager.optimizer_step(optimizer)

                do_validation = global_step % config.val_steps_freq == 0

                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq if global_step > 0 else running_loss
                    logger.log_metrics(
                        {"train/loss": avg_loss, "train/accuracy": accuracy},
                        commit=not do_validation,
                        step=global_step,
                    )
                    running_loss = 0.0

                # Validation phase
                if do_validation:
                    avg_val_loss, accuracy = validate(
                        model,
                        eval_dataloader,
                        loss_fn,
                        device_manager,
                        config,
                        logger,
                    )
                    model.train()

                    logger.log_metrics(
                        {"epoch": epoch + 1, "val/loss": avg_val_loss, "val/accuracy": accuracy},
                        step=global_step,
                    )

                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                global_step += 1

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model
        final_model_path = checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "test_vit_finetuning_stanfordcars.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Start training
    train(config, device_manager, logger, checkpoint_manager)
