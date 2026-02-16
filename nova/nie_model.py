"""
Neural Intent Engine — From-Scratch Transformer Classifier
==========================================================

Implements the full pipeline described in Steps 2–10:

  Step 2:  Tokenization       — split → vocab → numericise → pad
  Step 3:  Word Embeddings    — learnable dense vectors (embed_dim)
  Step 4:  Positional Encoding — learnable position embeddings
  Step 5:  Self-Attention     — Query / Key / Value dot-product attention
  Step 6:  Feed-Forward       — 2-layer MLP after attention
  Step 7:  Pooling            — mean-pool across the sequence
  Step 8:  Classification     — linear projection + softmax → probabilities
  Step 9:  Training           — cross-entropy loss + SGD with gradient descent
  Step 10: Inference          — instant offline intent prediction

Supports GPU acceleration via PyTorch CUDA when available.
Falls back to pure NumPy on CPU.
"""

import numpy as np
import json
import os
import re
from typing import List, Tuple, Dict, Optional

# ── GPU Auto-Detection (PyTorch CUDA) ───────────────────────────
USE_GPU = False
DEVICE = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        USE_GPU = True
except ImportError:
    torch = None


# ╔══════════════════════════════════════════════════════════════╗
# ║  Step 2: Tokenizer                                         ║
# ╚══════════════════════════════════════════════════════════════╝

class Tokenizer:
    """
    Breaks sentences into pieces the computer can understand.

    Input:  "lock my computer"
    Step 1: Split by spaces  → ["lock", "my", "computer"]
    Step 2: Build vocabulary → {lock: 1, my: 2, computer: 3}   (0 = PAD)
    Step 3: Convert to IDs   → [1, 2, 3]
    Step 4: Pad to max_len   → [1, 2, 3, 0, 0, 0, 0, 0]
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_len: int = 12):
        self.max_len = max_len
        self.word2id: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.id2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_size = 2  # PAD + UNK

    def _clean(self, text: str) -> List[str]:
        """Lowercase + remove punctuation + split on whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def build_vocab(self, sentences: List[str]):
        """Build vocabulary from a list of training sentences."""
        for sentence in sentences:
            for word in self._clean(sentence):
                if word not in self.word2id:
                    idx = self.vocab_size
                    self.word2id[word] = idx
                    self.id2word[idx] = word
                    self.vocab_size += 1

    def encode(self, sentence: str) -> np.ndarray:
        """Tokenise + pad a single sentence → fixed-length int array."""
        tokens = self._clean(sentence)
        ids = [self.word2id.get(t, self.word2id[self.UNK_TOKEN]) for t in tokens]
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int32)

    def encode_batch(self, sentences: List[str]) -> np.ndarray:
        return np.stack([self.encode(s) for s in sentences])

    def to_dict(self) -> dict:
        return {
            "max_len": self.max_len,
            "word2id": self.word2id,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        tok = cls(max_len=d["max_len"])
        tok.word2id = d["word2id"]
        tok.id2word = {int(v): k for k, v in d["word2id"].items()}
        tok.vocab_size = d["vocab_size"]
        return tok


# ╔══════════════════════════════════════════════════════════════╗
# ║  NumPy Utility Functions                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)

def cross_entropy_loss(probs, targets):
    """Works with both numpy arrays and torch tensors."""
    if torch is not None and isinstance(probs, torch.Tensor):
        batch_size = probs.shape[0]
        log_probs = -torch.log(probs[torch.arange(batch_size), targets] + 1e-9)
        return log_probs.mean().item()
    else:
        batch_size = probs.shape[0]
        log_probs = -np.log(probs[np.arange(batch_size), targets] + 1e-9)
        return float(log_probs.mean())

def he_init(shape: Tuple[int, ...]) -> np.ndarray:
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PyTorch GPU Model (used when CUDA is available)            ║
# ╚══════════════════════════════════════════════════════════════╝

if torch is not None:
    class NeuralIntentClassifierTorch(torch.nn.Module):
        """
        Same architecture as the NumPy version, but using PyTorch
        for GPU-accelerated training on NVIDIA GPUs.

        Architecture:
            text → [Tokenizer] → [Embedding + PosEmbed] → [Self-Attention]
            → [LayerNorm] → [FFN] → [LayerNorm] → [MeanPool] → [Classifier]
        """

        def __init__(self, vocab_size=200, max_len=12, embed_dim=64,
                     ff_dim=128, num_classes=5):
            super().__init__()
            self.max_len = max_len
            self.embed_dim = embed_dim

            # Step 3+4: Word + Position Embeddings
            self.word_embed = torch.nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = torch.nn.Embedding(max_len, embed_dim)

            # Step 5: Self-Attention (Q, K, V projections)
            self.Wq = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.Wk = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            self.Wv = torch.nn.Linear(embed_dim, embed_dim, bias=False)

            # LayerNorm 1
            self.ln1 = torch.nn.LayerNorm(embed_dim)

            # Step 6: Feed-Forward Network
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, ff_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_dim, embed_dim),
            )

            # LayerNorm 2
            self.ln2 = torch.nn.LayerNorm(embed_dim)

            # Dropout for regularization
            self.dropout = torch.nn.Dropout(0.1)

            # Step 8: Classification Head
            self.classifier = torch.nn.Linear(embed_dim, num_classes)

        def forward(self, token_ids):
            """
            token_ids: (batch, seq_len) LongTensor
            returns:   (batch, num_classes) probability tensor
            """
            B, S = token_ids.shape

            # Step 3+4: Embeddings
            positions = torch.arange(S, device=token_ids.device)
            x = self.word_embed(token_ids) + self.pos_embed(positions)

            # Step 5: Self-Attention
            Q = self.Wq(x)
            K = self.Wk(x)
            V = self.Wv(x)
            scale = self.embed_dim ** 0.5
            scores = torch.bmm(Q, K.transpose(1, 2)) / scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.bmm(attn_weights, V)

            # Residual + LayerNorm
            x = self.ln1(x + self.dropout(attn_out))

            # Step 6: Feed-Forward + Residual + LayerNorm
            ffn_out = self.ffn(x)
            x = self.ln2(x + self.dropout(ffn_out))

            # Step 7: Mean Pooling (mask PAD tokens)
            mask = (token_ids != 0).float().unsqueeze(-1)  # (B, S, 1)
            summed = (x * mask).sum(dim=1)                  # (B, D)
            counts = mask.sum(dim=1).clamp(min=1)            # (B, 1)
            pooled = summed / counts                         # (B, D)

            # Step 8: Classification
            logits = self.classifier(pooled)
            probs = torch.softmax(logits, dim=-1)
            return probs

        def predict(self, tokenizer, sentence):
            """
            Returns (predicted_class, confidence, all_probs_numpy).
            Runs on GPU if available.
            """
            self.eval()
            with torch.no_grad():
                ids = tokenizer.encode(sentence)
                ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                ids_tensor = ids_tensor.to(next(self.parameters()).device)
                probs = self.forward(ids_tensor)
                probs_np = probs.cpu().numpy()[0]
                pred = int(np.argmax(probs_np))
                conf = float(probs_np[pred])
            return pred, conf, probs_np

        def save(self, path: str):
            """Save weights as .npz for compatibility with NumPy version."""
            state = self.state_dict()
            np.savez_compressed(
                path,
                word_embed=state["word_embed.weight"].cpu().numpy(),
                pos_embed=state["pos_embed.weight"].cpu().numpy(),
                Wq=state["Wq.weight"].cpu().numpy().T,
                Wk=state["Wk.weight"].cpu().numpy().T,
                Wv=state["Wv.weight"].cpu().numpy().T,
                W1=state["ffn.0.weight"].cpu().numpy().T,
                b1=state["ffn.0.bias"].cpu().numpy(),
                W2=state["ffn.2.weight"].cpu().numpy().T,
                b2=state["ffn.2.bias"].cpu().numpy(),
                Wc=state["classifier.weight"].cpu().numpy().T,
                bc=state["classifier.bias"].cpu().numpy(),
                ln1_gamma=state["ln1.weight"].cpu().numpy(),
                ln1_beta=state["ln1.bias"].cpu().numpy(),
                ln2_gamma=state["ln2.weight"].cpu().numpy(),
                ln2_beta=state["ln2.bias"].cpu().numpy(),
            )

        def load(self, path: str):
            """Load weights from .npz (compatible with NumPy version)."""
            data = np.load(path)
            device = next(self.parameters()).device
            state = self.state_dict()
            state["word_embed.weight"] = torch.tensor(data["word_embed"], dtype=torch.float32, device=device)
            state["pos_embed.weight"] = torch.tensor(data["pos_embed"], dtype=torch.float32, device=device)
            state["Wq.weight"] = torch.tensor(data["Wq"].T, dtype=torch.float32, device=device)
            state["Wk.weight"] = torch.tensor(data["Wk"].T, dtype=torch.float32, device=device)
            state["Wv.weight"] = torch.tensor(data["Wv"].T, dtype=torch.float32, device=device)
            state["ffn.0.weight"] = torch.tensor(data["W1"].T, dtype=torch.float32, device=device)
            state["ffn.0.bias"] = torch.tensor(data["b1"], dtype=torch.float32, device=device)
            state["ffn.2.weight"] = torch.tensor(data["W2"].T, dtype=torch.float32, device=device)
            state["ffn.2.bias"] = torch.tensor(data["b2"], dtype=torch.float32, device=device)
            state["classifier.weight"] = torch.tensor(data["Wc"].T, dtype=torch.float32, device=device)
            state["classifier.bias"] = torch.tensor(data["bc"], dtype=torch.float32, device=device)
            state["ln1.weight"] = torch.tensor(data["ln1_gamma"], dtype=torch.float32, device=device)
            state["ln1.bias"] = torch.tensor(data["ln1_beta"], dtype=torch.float32, device=device)
            state["ln2.weight"] = torch.tensor(data["ln2_gamma"], dtype=torch.float32, device=device)
            state["ln2.bias"] = torch.tensor(data["ln2_beta"], dtype=torch.float32, device=device)
            self.load_state_dict(state)


# ╔══════════════════════════════════════════════════════════════╗
# ║  NumPy CPU Model (fallback when PyTorch unavailable)        ║
# ╚══════════════════════════════════════════════════════════════╝

class EmbeddingLayer:
    def __init__(self, vocab_size, max_len, embed_dim):
        self.embed_dim = embed_dim
        self.word_embed = np.random.randn(vocab_size, embed_dim) * 0.05
        self.pos_embed = np.random.randn(max_len, embed_dim) * 0.05
        self.grad_word_embed = np.zeros_like(self.word_embed)
        self.grad_pos_embed = np.zeros_like(self.pos_embed)

    def forward(self, token_ids):
        self.token_ids = token_ids
        batch, seq_len = token_ids.shape
        word_out = self.word_embed[token_ids]
        pos_out = self.pos_embed[np.arange(seq_len)]
        self.output = word_out + pos_out
        return self.output

    def backward(self, grad_output):
        batch, seq_len, _ = grad_output.shape
        self.grad_pos_embed[:seq_len] += grad_output.sum(axis=0)
        np.add.at(self.grad_word_embed, self.token_ids, grad_output)

    def zero_grad(self):
        self.grad_word_embed[:] = 0
        self.grad_pos_embed[:] = 0

    def step(self, lr):
        self.word_embed -= lr * self.grad_word_embed
        self.pos_embed -= lr * self.grad_pos_embed


class SelfAttention:
    def __init__(self, embed_dim):
        self.d = embed_dim
        scale = np.sqrt(2.0 / embed_dim)
        self.Wq = np.random.randn(embed_dim, embed_dim) * scale
        self.Wk = np.random.randn(embed_dim, embed_dim) * scale
        self.Wv = np.random.randn(embed_dim, embed_dim) * scale
        self.grad_Wq = np.zeros_like(self.Wq)
        self.grad_Wk = np.zeros_like(self.Wk)
        self.grad_Wv = np.zeros_like(self.Wv)

    def forward(self, x):
        self.x = x
        B, S, D = x.shape
        self.Q = x @ self.Wq
        self.K = x @ self.Wk
        self.V = x @ self.Wv
        scale = np.sqrt(D)
        self.scores = (self.Q @ self.K.transpose(0, 2, 1)) / scale
        self.attn_weights = softmax(self.scores)
        self.output = self.attn_weights @ self.V
        return self.output

    def backward(self, grad_output):
        B, S, D = grad_output.shape
        scale = np.sqrt(D)
        grad_V = self.attn_weights.transpose(0, 2, 1) @ grad_output
        grad_attn = grad_output @ self.V.transpose(0, 2, 1)
        sum_term = (grad_attn * self.attn_weights).sum(axis=-1, keepdims=True)
        grad_scores = self.attn_weights * (grad_attn - sum_term)
        grad_scores /= scale
        grad_Q = grad_scores @ self.K
        grad_K = grad_scores.transpose(0, 2, 1) @ self.Q
        self.grad_Wq += np.einsum("bsi,bsj->ij", self.x, grad_Q)
        self.grad_Wk += np.einsum("bsi,bsj->ij", self.x, grad_K)
        self.grad_Wv += np.einsum("bsi,bsj->ij", self.x, grad_V)
        grad_x = grad_Q @ self.Wq.T + grad_K @ self.Wk.T + grad_V @ self.Wv.T
        return grad_x

    def zero_grad(self):
        self.grad_Wq[:] = 0; self.grad_Wk[:] = 0; self.grad_Wv[:] = 0

    def step(self, lr):
        self.Wq -= lr * self.grad_Wq
        self.Wk -= lr * self.grad_Wk
        self.Wv -= lr * self.grad_Wv


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = he_init((embed_dim, ff_dim))
        self.b1 = np.zeros(ff_dim)
        self.W2 = he_init((ff_dim, embed_dim))
        self.b2 = np.zeros(embed_dim)
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, grad_output):
        self.grad_W2 += np.einsum("bsi,bsj->ij", self.a1, grad_output)
        self.grad_b2 += grad_output.sum(axis=(0, 1))
        grad_a1 = grad_output @ self.W2.T
        grad_z1 = grad_a1 * relu_grad(self.z1)
        self.grad_W1 += np.einsum("bsi,bsj->ij", self.x, grad_z1)
        self.grad_b1 += grad_z1.sum(axis=(0, 1))
        return grad_z1 @ self.W1.T

    def zero_grad(self):
        self.grad_W1[:] = 0; self.grad_b1[:] = 0
        self.grad_W2[:] = 0; self.grad_b2[:] = 0

    def step(self, lr):
        self.W1 -= lr * self.grad_W1; self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2; self.b2 -= lr * self.grad_b2


class ClassificationHead:
    def __init__(self, embed_dim, num_classes):
        self.W = he_init((embed_dim, num_classes))
        self.b = np.zeros(num_classes)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        logits = x @ self.W + self.b
        self.probs = softmax(logits)
        return self.probs

    def backward(self, targets):
        B = targets.shape[0]
        grad_logits = self.probs.copy()
        grad_logits[np.arange(B), targets] -= 1
        grad_logits /= B
        self.grad_W += self.x.T @ grad_logits
        self.grad_b += grad_logits.sum(axis=0)
        return grad_logits @ self.W.T

    def zero_grad(self):
        self.grad_W[:] = 0; self.grad_b[:] = 0

    def step(self, lr):
        self.W -= lr * self.grad_W; self.b -= lr * self.grad_b


class NeuralIntentClassifier:
    """
    NumPy CPU fallback — used when PyTorch is not available.
    Same architecture as the PyTorch version.
    """

    def __init__(self, vocab_size=200, max_len=12, embed_dim=64,
                 ff_dim=128, num_classes=5):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.embedding = EmbeddingLayer(vocab_size, max_len, embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.classifier = ClassificationHead(embed_dim, num_classes)
        self.ln1_gamma = np.ones(embed_dim)
        self.ln1_beta = np.zeros(embed_dim)
        self.ln2_gamma = np.ones(embed_dim)
        self.ln2_beta = np.zeros(embed_dim)

    @staticmethod
    def _layer_norm(x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, x_norm, mean, var

    def forward(self, token_ids):
        x = self.embedding.forward(token_ids)
        attn_out = self.attention.forward(x)
        x_res1 = x + attn_out
        self.ln1_out, self.ln1_xn, self.ln1_m, self.ln1_v = \
            self._layer_norm(x_res1, self.ln1_gamma, self.ln1_beta)
        ffn_out = self.ffn.forward(self.ln1_out)
        x_res2 = self.ln1_out + ffn_out
        self.ln2_out, self.ln2_xn, self.ln2_m, self.ln2_v = \
            self._layer_norm(x_res2, self.ln2_gamma, self.ln2_beta)
        mask = (token_ids != 0).astype(np.float64)
        mask_expanded = mask[:, :, np.newaxis]
        summed = (self.ln2_out * mask_expanded).sum(axis=1)
        counts = mask.sum(axis=1, keepdims=True).clip(min=1)
        self.pooled = summed / counts
        self._pool_mask = mask_expanded
        self._pool_counts = counts
        probs = self.classifier.forward(self.pooled)
        return probs

    def backward(self, token_ids, targets):
        B, S = token_ids.shape
        grad_pooled = self.classifier.backward(targets)
        grad_ln2 = (grad_pooled[:, np.newaxis, :] / self._pool_counts[:, :, np.newaxis]) \
                    * self._pool_mask
        grad_res2 = grad_ln2 * self.ln2_gamma
        grad_ffn = self.ffn.backward(grad_res2)
        grad_ln1 = grad_res2 + grad_ffn
        grad_res1 = grad_ln1 * self.ln1_gamma
        grad_attn = self.attention.backward(grad_res1)
        grad_embed = grad_res1 + grad_attn
        self.embedding.backward(grad_embed)

    def zero_grad(self):
        self.embedding.zero_grad()
        self.attention.zero_grad()
        self.ffn.zero_grad()
        self.classifier.zero_grad()

    def step(self, lr):
        self.embedding.step(lr)
        self.attention.step(lr)
        self.ffn.step(lr)
        self.classifier.step(lr)

    def predict(self, tokenizer, sentence):
        ids = tokenizer.encode(sentence)[np.newaxis, :]
        probs = self.forward(ids)
        pred = int(np.argmax(probs, axis=1)[0])
        conf = float(probs[0, pred])
        return pred, conf, probs[0]

    def save(self, path):
        np.savez_compressed(
            path,
            word_embed=self.embedding.word_embed,
            pos_embed=self.embedding.pos_embed,
            Wq=self.attention.Wq, Wk=self.attention.Wk, Wv=self.attention.Wv,
            W1=self.ffn.W1, b1=self.ffn.b1, W2=self.ffn.W2, b2=self.ffn.b2,
            Wc=self.classifier.W, bc=self.classifier.b,
            ln1_gamma=self.ln1_gamma, ln1_beta=self.ln1_beta,
            ln2_gamma=self.ln2_gamma, ln2_beta=self.ln2_beta,
        )

    def load(self, path):
        data = np.load(path)
        self.embedding.word_embed = data["word_embed"]
        self.embedding.pos_embed = data["pos_embed"]
        self.attention.Wq = data["Wq"]
        self.attention.Wk = data["Wk"]
        self.attention.Wv = data["Wv"]
        self.ffn.W1 = data["W1"]
        self.ffn.b1 = data["b1"]
        self.ffn.W2 = data["W2"]
        self.ffn.b2 = data["b2"]
        self.classifier.W = data["Wc"]
        self.classifier.b = data["bc"]
        self.ln1_gamma = data["ln1_gamma"]
        self.ln1_beta = data["ln1_beta"]
        self.ln2_gamma = data["ln2_gamma"]
        self.ln2_beta = data["ln2_beta"]


# ╔══════════════════════════════════════════════════════════════╗
# ║  Factory: Auto-select GPU (PyTorch) or CPU (NumPy) model    ║
# ╚══════════════════════════════════════════════════════════════╝

def create_model(vocab_size=200, max_len=12, embed_dim=64,
                 ff_dim=128, num_classes=5):
    """Create the best available model (GPU PyTorch or CPU NumPy)."""
    if USE_GPU and torch is not None:
        model = NeuralIntentClassifierTorch(
            vocab_size=vocab_size, max_len=max_len,
            embed_dim=embed_dim, ff_dim=ff_dim, num_classes=num_classes
        )
        model = model.to(DEVICE)
        return model
    else:
        return NeuralIntentClassifier(
            vocab_size=vocab_size, max_len=max_len,
            embed_dim=embed_dim, ff_dim=ff_dim, num_classes=num_classes
        )
