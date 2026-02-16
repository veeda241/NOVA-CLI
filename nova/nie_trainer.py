"""
NIE Trainer — Train the Neural Intent Classifier
=================================================

This script:
  1. Loads training data from nie_data.py
  2. Builds vocabulary (tokenisation)
  3. Trains the network using gradient descent
  4. Evaluates on validation data
  5. Saves the trained model + tokenizer to disk

Usage:
    python -m nova.nie_trainer          (from project root)
    python nova/nie_trainer.py          (directly)
"""

import os
import sys
import json
import time
import io
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure the parent directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nova.nie_model import (
    NeuralIntentClassifier, Tokenizer, cross_entropy_loss,
    USE_GPU, DEVICE, create_model
)
from nova.nie_data import TRAINING_DATA, VALIDATION_DATA, INTENT_LABELS, NUM_INTENTS

# Try torch import
try:
    import torch
except ImportError:
    torch = None

# ──────────────────────────
# Configuration
# ──────────────────────────
EMBED_DIM   = 64
FF_DIM      = 128
MAX_LEN     = 12
EPOCHS      = 1500
BATCH_SIZE  = 16
LR_INIT     = 0.01
LR_MIN      = 0.0003
SEED        = 42

# Where to save the model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nie_weights")


def train():
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)

    print("=" * 65)
    print("  [BRAIN]  Neural Intent Engine - Training Pipeline")
    print("=" * 65)
    if USE_GPU and torch is not None:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [GPU] CUDA enabled! Device: {gpu_name}")
        print(f"  [GPU] VRAM: {gpu_mem:.1f} GB | PyTorch {torch.__version__}")
    else:
        print("  [CPU] Training on CPU (numpy).")
    print()

    # ─────────────────────────────────────────────
    # Step 2: Tokenization — Build vocabulary
    # ─────────────────────────────────────────────
    print("[STEP 2] Building vocabulary (Tokenization)...")
    sentences_train = [s for s, _ in TRAINING_DATA]
    labels_train    = np.array([l for _, l in TRAINING_DATA], dtype=np.int32)

    sentences_val = [s for s, _ in VALIDATION_DATA]
    labels_val    = np.array([l for _, l in VALIDATION_DATA], dtype=np.int32)

    tokenizer = Tokenizer(max_len=MAX_LEN)
    tokenizer.build_vocab(sentences_train + sentences_val)

    print(f"   Vocabulary size : {tokenizer.vocab_size}")
    print(f"   Max sequence len: {MAX_LEN}")
    print(f"   Training samples: {len(sentences_train)}")
    print(f"   Validation samples: {len(sentences_val)}")

    # Show tokenisation example
    example = "lock my computer"
    tokens = tokenizer._clean(example)
    ids = tokenizer.encode(example)
    print(f'\n   Example: "{example}"')
    print(f"   Tokens : {tokens}")
    print(f"   IDs    : {ids.tolist()}")
    print()

    # Encode all data
    X_train = tokenizer.encode_batch(sentences_train)
    X_val   = tokenizer.encode_batch(sentences_val)

    # ─────────────────────────────────────────────
    # Steps 3–8: Build the neural network
    # ─────────────────────────────────────────────
    print("[STEPS 3-8] Building Neural Intent Classifier...")
    print(f"   Embed dim : {EMBED_DIM}")
    print(f"   FF dim    : {FF_DIM}")
    print(f"   Classes   : {NUM_INTENTS}  {list(INTENT_LABELS.values())}")
    print()

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        ff_dim=FF_DIM,
        num_classes=NUM_INTENTS,
    )

    # ─────────────────────────────────────────────
    # Step 9: Training loop
    # ─────────────────────────────────────────────
    print("[STEP 9] Training with Gradient Descent...")
    print(f"   Epochs        : {EPOCHS}")
    print(f"   Batch size    : {BATCH_SIZE}")
    print(f"   Learning rate : {LR_INIT} -> {LR_MIN} (cosine decay)")
    print("-" * 65)

    n_train = len(sentences_train)
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    if USE_GPU and torch is not None:
        # ── PyTorch GPU Training ──
        # Init weights properly for PyTorch
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

        optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR_MIN
        )

        # Move data to GPU
        X_train_t = torch.tensor(X_train, dtype=torch.long, device=DEVICE)
        y_train_t = torch.tensor(labels_train, dtype=torch.long, device=DEVICE)
        X_val_t = torch.tensor(X_val, dtype=torch.long, device=DEVICE)
        y_val_t = torch.tensor(labels_val, dtype=torch.long, device=DEVICE)

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(1, EPOCHS + 1):
            model.train()

            # Shuffle
            perm = torch.randperm(n_train, device=DEVICE)
            X_shuffled = X_train_t[perm]
            y_shuffled = y_train_t[perm]

            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0

            for i in range(0, n_train, BATCH_SIZE):
                xb = X_shuffled[i:i+BATCH_SIZE]
                yb = y_shuffled[i:i+BATCH_SIZE]

                optimizer.zero_grad()
                probs = model(xb)

                # Use CrossEntropyLoss on logits (recompute from probs)
                # Actually, probs already has softmax. Use NLL-like loss:
                loss = -torch.log(probs[torch.arange(len(yb)), yb] + 1e-9).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = probs.argmax(dim=1)
                epoch_correct += (preds == yb).sum().item()
                n_batches += 1

            scheduler.step()

            train_acc = epoch_correct / n_train * 100
            avg_loss = epoch_loss / n_batches

            # Validation
            model.eval()
            with torch.no_grad():
                val_probs = model(X_val_t)
                val_preds = val_probs.argmax(dim=1)
                val_acc = (val_preds == y_val_t).sum().item() / len(labels_val) * 100

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                os.makedirs(MODEL_DIR, exist_ok=True)
                model.save(os.path.join(MODEL_DIR, "nie_model.npz"))

            # Print progress
            elapsed = time.time() - start_time
            if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
                bar_len = 20
                filled = int(bar_len * train_acc / 100)
                bar = "#" * filled + "." * (bar_len - filled)
                lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                      f"Train: {train_acc:5.1f}% [{bar}] | "
                      f"Val: {val_acc:5.1f}% | LR: {lr:.5f} | {elapsed:.1f}s")

    else:
        # ── NumPy CPU Training ──
        for epoch in range(1, EPOCHS + 1):
            lr = LR_MIN + 0.5 * (LR_INIT - LR_MIN) * (1 + np.cos(np.pi * epoch / EPOCHS))

            perm = np.random.permutation(n_train)
            X_shuffled = X_train[perm]
            y_shuffled = labels_train[perm]

            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0

            for i in range(0, n_train, BATCH_SIZE):
                xb = X_shuffled[i:i+BATCH_SIZE]
                yb = y_shuffled[i:i+BATCH_SIZE]

                model.zero_grad()
                probs = model.forward(xb)
                loss = cross_entropy_loss(probs, yb)
                epoch_loss += loss

                preds = np.argmax(probs, axis=1)
                epoch_correct += (preds == yb).sum()

                model.backward(xb, yb)
                model.step(lr)
                n_batches += 1

            train_acc = epoch_correct / n_train * 100
            avg_loss = epoch_loss / n_batches

            val_probs = model.forward(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = (val_preds == labels_val).sum() / len(labels_val) * 100

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                os.makedirs(MODEL_DIR, exist_ok=True)
                model.save(os.path.join(MODEL_DIR, "nie_model.npz"))

            elapsed = time.time() - start_time
            if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
                bar_len = 20
                filled = int(bar_len * train_acc / 100)
                bar = "#" * filled + "." * (bar_len - filled)
                print(f"   Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                      f"Train: {train_acc:5.1f}% [{bar}] | "
                      f"Val: {val_acc:5.1f}% | LR: {lr:.5f} | {elapsed:.1f}s")

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"\n[OK] Training complete in {total_time:.1f}s")
    print(f"   Best validation accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")
    print(f"   Model saved to: {MODEL_DIR}/")

    # Save tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w") as f:
        json.dump(tokenizer.to_dict(), f)

    # ─────────────────────────────────────────────
    # Step 10: Inference Test
    # ─────────────────────────────────────────────
    print(f"\n[STEP 10] Testing Instant Inference...")
    print("-" * 65)

    # Reload best model for testing
    test_model = create_model(
        vocab_size=tokenizer.vocab_size,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        ff_dim=FF_DIM,
        num_classes=NUM_INTENTS,
    )
    test_model.load(os.path.join(MODEL_DIR, "nie_model.npz"))

    test_sentences = [
        "lock my computer", "secure the laptop",
        "increase volume", "make it louder",
        "decrease volume", "lower the volume",
        "set volume to 50", "volume 75",
        "what's my battery", "check cpu usage",
        "open chrome", "launch notepad",
        "close chrome", "kill firefox",
        "take a screenshot", "capture the screen",
        "increase brightness", "make it brighter",
        "set brightness to 80", "brightness 60",
        "decrease brightness", "dim the screen",
        "tell me a joke", "what's the weather",
        "hello there", "system status",
        "open calculator", "close task manager",
    ]

    for sent in test_sentences:
        t0 = time.perf_counter()
        pred_id, conf, _ = test_model.predict(tokenizer, sent)
        ms = (time.perf_counter() - t0) * 1000
        label = INTENT_LABELS.get(pred_id, "???")

        bar_len = 20
        filled = int(bar_len * conf)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f'   "{sent}"')
        print(f"      -> {label:15s} [{bar}] {conf*100:5.1f}%  ({ms:.2f} ms)")
        print()

    print("=" * 65)
    print("  [DONE] Neural Intent Engine ready for integration!")
    print("=" * 65)


if __name__ == "__main__":
    train()
