# ============================================================
# TinyGPT v2
# ============================================================
# From-scratch GPT-style language model
#
# Allowed:
#   NumPy
#   SciPy
#   scikit-learn
#   pandas
#   matplotlib
#
# NOT USED:
#   TensorFlow
#   PyTorch
#   OpenCV
#
# Designed for Pydroid / Android.
#
# Features:
#   - Byte-level UTF-8 tokenizer
#   - Token embeddings
#   - Positional embeddings
#   - Causal self-attention
#   - Feed-forward network
#   - Layer normalization
#   - Residual connections
#   - Softmax + cross entropy
#   - Manual backpropagation
#   - Adam optimizer
#   - Persistent model
#   - Persistent training examples
#   - Interactive teaching
#
# ============================================================

import os
import json
import difflib
import re
import contextlib
import io
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "tinygpt_v2_model.npz"
DATA_FILE = "tinygpt_v2_training.jsonl"
KNOWLEDGE_FILE = "tinygpt_v2_knowledge.jsonl"
AUTO_TRAIN_PENDING = True
AUTO_TRAIN_STEPS = 1

BLOCK_SIZE = 32
EMBED_SIZE = 32
HIDDEN_SIZE = 64

LEARNING_RATE = 0.003
TRAIN_STEPS = 20

MAX_GENERATION = 80
TEMPERATURE = 0.8
USE_NEURAL_FALLBACK = False

SEED = 42

np.random.seed(SEED)


# ============================================================
# STARTER CORPUS
# ============================================================

STARTER_TEXT = """
Electronics engineering is the study of electronic circuits,
devices, signals and systems.

A resistor limits current in a circuit.
A capacitor stores energy in an electric field.
An inductor stores energy in a magnetic field.
A diode allows current to flow mainly in one direction.
A transistor can operate as a switch or amplifier.

Digital electronics uses binary signals.
Binary numbers use zero and one.
Logic gates include AND, OR, NOT, NAND, NOR, XOR and XNOR.

Voltage is electrical potential difference.
Current is the flow of electric charge.
Resistance opposes current.
Ohm's law states that V equals I multiplied by R.

Frequency is the number of cycles per second.
The unit of frequency is hertz.

A signal can be analog or digital.
Sampling converts a continuous signal into discrete measurements.

Artificial intelligence allows computers to perform tasks
that normally require human intelligence.

Machine learning allows a model to learn patterns from data.
A neural network contains layers of mathematical operations.

A language model predicts the next token from previous tokens.
A transformer uses attention to process relationships between tokens.
Attention allows one token to consider other tokens in its context.
GPT means Generative Pretrained Transformer.

A neural network learns by changing its parameters.
Training uses a loss function and an optimization algorithm.
Gradient descent changes parameters in the direction that reduces loss.
Adam is an optimization algorithm based on gradient information.
"""


# ============================================================
# BYTE TOKENIZER
# ============================================================

class ByteTokenizer:

    def __init__(self):

        # Every possible byte is a token.
        self.vocab_size = 256

    def encode(self, text):

        return list(
            text.encode(
                "utf-8",
                errors="replace"
            )
        )

    def decode(self, tokens):

        values = []

        for token in tokens:

            token = int(token)

            if 0 <= token <= 255:
                values.append(token)

        try:

            return bytes(values).decode(
                "utf-8",
                errors="replace"
            )

        except Exception:

            return ""


# ============================================================
# MATH FUNCTIONS
# ============================================================

def softmax(x):

    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / (
        np.sum(
            exp_x,
            axis=-1,
            keepdims=True
        ) + 1e-12
    )


def gelu(x):

    return 0.5 * x * (
        1.0 +
        np.tanh(
            np.sqrt(2.0 / np.pi)
            *
            (
                x +
                0.044715 * x ** 3
            )
        )
    )


def gelu_derivative(x):

    t = np.sqrt(
        2.0 / np.pi
    ) * (
        x +
        0.044715 * x ** 3
    )

    tanh_t = np.tanh(t)

    dt = np.sqrt(
        2.0 / np.pi
    ) * (
        1.0 +
        3.0 *
        0.044715 *
        x ** 2
    )

    return (
        0.5 *
        (
            1.0 +
            tanh_t
        )
        +
        0.5 *
        x *
        (
            1.0 -
            tanh_t ** 2
        ) *
        dt
    )


# ============================================================
# LAYER NORMALIZATION
# ============================================================

def layer_norm_forward(x, eps=1e-5):

    mean = np.mean(
        x,
        axis=-1,
        keepdims=True
    )

    variance = np.mean(
        (
            x - mean
        ) ** 2,
        axis=-1,
        keepdims=True
    )

    inv_std = 1.0 / np.sqrt(
        variance + eps
    )

    normalized = (
        x - mean
    ) * inv_std

    cache = (
        normalized,
        inv_std
    )

    return normalized, cache


def layer_norm_backward(
    grad,
    cache
):

    normalized, inv_std = cache

    n = grad.shape[-1]

    grad_mean = np.mean(
        grad,
        axis=-1,
        keepdims=True
    )

    grad_norm_mean = np.mean(
        grad * normalized,
        axis=-1,
        keepdims=True
    )

    dx = (
        inv_std
        *
        (
            grad
            -
            grad_mean
            -
            normalized *
            grad_norm_mean
        )
    )

    return dx


# ============================================================
# ADAM
# ============================================================

class Adam:

    def __init__(
        self,
        model,
        learning_rate=LEARNING_RATE
    ):

        self.lr = learning_rate

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self.t = 0

        self.m = {}
        self.v = {}

        for name in model.parameter_names():

            parameter = getattr(
                model,
                name
            )

            self.m[name] = np.zeros_like(
                parameter
            )

            self.v[name] = np.zeros_like(
                parameter
            )

    def step(
        self,
        model,
        gradients
    ):

        self.t += 1

        for name in model.parameter_names():

            parameter = getattr(
                model,
                name
            )

            gradient = gradients[name]

            self.m[name] = (
                self.beta1 *
                self.m[name]
                +
                (1.0 - self.beta1)
                *
                gradient
            )

            self.v[name] = (
                self.beta2 *
                self.v[name]
                +
                (1.0 - self.beta2)
                *
                gradient ** 2
            )

            m_hat = (
                self.m[name]
                /
                (
                    1.0 -
                    self.beta1 ** self.t
                )
            )

            v_hat = (
                self.v[name]
                /
                (
                    1.0 -
                    self.beta2 ** self.t
                )
            )

            parameter -= (
                self.lr *
                m_hat
                /
                (
                    np.sqrt(v_hat)
                    +
                    self.epsilon
                )
            )


# ============================================================
# TINYGPT
# ============================================================

class TinyGPT:

    def __init__(
        self,
        vocab_size=256
    ):

        self.vocab_size = vocab_size

        self.block_size = BLOCK_SIZE
        self.embed_size = EMBED_SIZE
        self.hidden_size = HIDDEN_SIZE

        scale = 0.02

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.token_embedding = (
            np.random.randn(
                vocab_size,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        self.position_embedding = (
            np.random.randn(
                BLOCK_SIZE,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.Wq = (
            np.random.randn(
                EMBED_SIZE,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        self.Wk = (
            np.random.randn(
                EMBED_SIZE,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        self.Wv = (
            np.random.randn(
                EMBED_SIZE,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        self.Wo = (
            np.random.randn(
                EMBED_SIZE,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        self.W1 = (
            np.random.randn(
                EMBED_SIZE,
                HIDDEN_SIZE
            ).astype(np.float32)
            * scale
        )

        self.b1 = np.zeros(
            HIDDEN_SIZE,
            dtype=np.float32
        )

        self.W2 = (
            np.random.randn(
                HIDDEN_SIZE,
                EMBED_SIZE
            ).astype(np.float32)
            * scale
        )

        self.b2 = np.zeros(
            EMBED_SIZE,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Output head
        # ----------------------------------------------------

        self.W_vocab = (
            np.random.randn(
                EMBED_SIZE,
                vocab_size
            ).astype(np.float32)
            * scale
        )

        self.b_vocab = np.zeros(
            vocab_size,
            dtype=np.float32
        )

    # ========================================================
    # PARAMETERS
    # ========================================================

    def parameter_names(self):

        return [
            "token_embedding",
            "position_embedding",
            "Wq",
            "Wk",
            "Wv",
            "Wo",
            "W1",
            "b1",
            "W2",
            "b2",
            "W_vocab",
            "b_vocab"
        ]

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        token_ids,
        training=True
    ):

        token_ids = np.asarray(
            token_ids,
            dtype=np.int64
        )

        if len(token_ids) > self.block_size:

            token_ids = token_ids[
                -self.block_size:
            ]

        T = len(token_ids)

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        token_x = self.token_embedding[
            token_ids
        ]

        position_x = (
            self.position_embedding[
                :T
            ]
        )

        x = (
            token_x +
            position_x
        )

        # ----------------------------------------------------
        # First LayerNorm
        # ----------------------------------------------------

        norm1, ln1_cache = (
            layer_norm_forward(x)
        )

        # ----------------------------------------------------
        # Q K V
        # ----------------------------------------------------

        q = norm1 @ self.Wq
        k = norm1 @ self.Wk
        v = norm1 @ self.Wv

        # ----------------------------------------------------
        # Causal attention
        # ----------------------------------------------------

        scale = np.sqrt(
            self.embed_size
        )

        scores = (
            q @ k.T
        ) / scale

        causal_mask = np.triu(
            np.ones(
                (T, T),
                dtype=bool
            ),
            k=1
        )

        scores = np.where(
            causal_mask,
            -1e9,
            scores
        )

        attention = softmax(
            scores
        )

        attention_output = (
            attention @ v
        )

        attention_projected = (
            attention_output @ self.Wo
        )

        residual1 = (
            x +
            attention_projected
        )

        # ----------------------------------------------------
        # Second LayerNorm
        # ----------------------------------------------------

        norm2, ln2_cache = (
            layer_norm_forward(
                residual1
            )
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        hidden_pre = (
            norm2 @ self.W1
            +
            self.b1
        )

        hidden = gelu(
            hidden_pre
        )

        ff = (
            hidden @ self.W2
            +
            self.b2
        )

        residual2 = (
            residual1 +
            ff
        )

        # ----------------------------------------------------
        # Third LayerNorm
        # ----------------------------------------------------

        norm3, ln3_cache = (
            layer_norm_forward(
                residual2
            )
        )

        # ----------------------------------------------------
        # Vocabulary logits
        # ----------------------------------------------------

        logits = (
            norm3 @ self.W_vocab
            +
            self.b_vocab
        )

        if not training:

            return logits

        cache = {
            "token_ids": token_ids,
            "x": x,
            "ln1": ln1_cache,
            "norm1": norm1,
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "attention": attention,
            "attention_output": attention_output,
            "residual1": residual1,
            "ln2": ln2_cache,
            "norm2": norm2,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
            "residual2": residual2,
            "ln3": ln3_cache,
            "norm3": norm3
        }

        return logits, cache

    # ========================================================
    # LOSS
    # ========================================================

    def loss(
        self,
        logits,
        targets
    ):

        probabilities = softmax(
            logits
        )

        probabilities = np.clip(
            probabilities,
            1e-9,
            1.0
        )

        selected = probabilities[
            np.arange(
                len(targets)
            ),
            targets
        ]

        return float(
            -np.mean(
                np.log(selected)
            )
        )

    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    def backward(
        self,
        logits,
        targets,
        cache
    ):

        grads = {}

        T = len(targets)

        # ----------------------------------------------------
        # Softmax + Cross Entropy
        # ----------------------------------------------------

        probabilities = softmax(
            logits
        )

        dlogits = probabilities.copy()

        dlogits[
            np.arange(T),
            targets
        ] -= 1.0

        dlogits /= T

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        grads["W_vocab"] = (
            cache["norm3"].T
            @
            dlogits
        )

        grads["b_vocab"] = (
            np.sum(
                dlogits,
                axis=0
            )
        )

        dnorm3 = (
            dlogits
            @
            self.W_vocab.T
        )

        # ----------------------------------------------------
        # LayerNorm 3
        # ----------------------------------------------------

        dresidual2 = (
            layer_norm_backward(
                dnorm3,
                cache["ln3"]
            )
        )

        # ----------------------------------------------------
        # Residual 2
        # ----------------------------------------------------

        dresidual1 = dresidual2.copy()

        dff = dresidual2

        # ----------------------------------------------------
        # Feed-forward
        # ----------------------------------------------------

        grads["W2"] = (
            cache["hidden"].T
            @
            dff
        )

        grads["b2"] = (
            np.sum(
                dff,
                axis=0
            )
        )

        dhidden = (
            dff
            @
            self.W2.T
        )

        dhidden_pre = (
            dhidden
            *
            gelu_derivative(
                cache["hidden_pre"]
            )
        )

        grads["W1"] = (
            cache["norm2"].T
            @
            dhidden_pre
        )

        grads["b1"] = (
            np.sum(
                dhidden_pre,
                axis=0
            )
        )

        dnorm2 = (
            dhidden_pre
            @
            self.W1.T
        )

        # ----------------------------------------------------
        # LayerNorm 2
        # ----------------------------------------------------

        dresidual1 += (
            layer_norm_backward(
                dnorm2,
                cache["ln2"]
            )
        )

        # ----------------------------------------------------
        # Attention residual
        # ----------------------------------------------------

        dattention_projected = (
            dresidual1
        )

        grads["Wo"] = (
            cache[
                "attention_output"
            ].T
            @
            dattention_projected
        )

        dattention_output = (
            dattention_projected
            @
            self.Wo.T
        )

        # ----------------------------------------------------
        # attention = softmax(scores) @ v
        # ----------------------------------------------------

        attention = cache[
            "attention"
        ]

        v = cache["v"]

        dattention = (
            dattention_output
            @
            v.T
        )

        dv = (
            attention.T
            @
            dattention_output
        )

        # ----------------------------------------------------
        # Softmax backward for attention
        # ----------------------------------------------------

        dscores = (
            attention
            *
            (
                dattention
                -
                np.sum(
                    dattention
                    *
                    attention,
                    axis=1,
                    keepdims=True
                )
            )
        )

        # Masked positions have no gradient.
        causal_mask = np.triu(
            np.ones(
                dscores.shape,
                dtype=bool
            ),
            k=1
        )

        dscores[
            causal_mask
        ] = 0.0

        scale = np.sqrt(
            self.embed_size
        )

        dq = (
            dscores
            @
            cache["k"]
            / scale
        )

        dk = (
            dscores.T
            @
            cache["q"]
            / scale
        )

        # ----------------------------------------------------
        # Q/K/V projections
        # ----------------------------------------------------

        norm1 = cache["norm1"]

        grads["Wq"] = (
            norm1.T
            @
            dq
        )

        grads["Wk"] = (
            norm1.T
            @
            dk
        )

        grads["Wv"] = (
            norm1.T
            @
            dv
        )

        dnorm1 = (
            dq @ self.Wq.T
            +
            dk @ self.Wk.T
            +
            dv @ self.Wv.T
        )

        # ----------------------------------------------------
        # LayerNorm 1
        # ----------------------------------------------------

        dx = layer_norm_backward(
            dnorm1,
            cache["ln1"]
        )

        # ----------------------------------------------------
        # Residual path
        # ----------------------------------------------------

        dx += dresidual1

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        grads["token_embedding"] = (
            np.zeros_like(
                self.token_embedding
            )
        )

        for i, token in enumerate(
            cache["token_ids"]
        ):

            grads[
                "token_embedding"
            ][token] += dx[i]

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------

        grads[
            "position_embedding"
        ] = np.zeros_like(
            self.position_embedding
        )

        for i in range(
            len(cache["token_ids"])
        ):

            grads[
                "position_embedding"
            ][i] += dx[i]

        return grads

    # ========================================================
    # TRAIN ONE EXAMPLE
    # ========================================================

    def train_example(
        self,
        tokens,
        optimizer
    ):

        if len(tokens) < 2:

            return 0.0

        tokens = tokens[
            :self.block_size + 1
        ]

        inputs = tokens[:-1]
        targets = tokens[1:]

        logits, cache = self.forward(
            inputs,
            training=True
        )

        loss = self.loss(
            logits,
            targets
        )

        gradients = self.backward(
            logits,
            targets,
            cache
        )

        # Gradient clipping.
        for name in gradients:

            gradients[name] = np.clip(
                gradients[name],
                -1.0,
                1.0
            )

        optimizer.step(
            self,
            gradients
        )

        return loss

    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        tokenizer,
        prompt,
        max_new_tokens=MAX_GENERATION,
        temperature=TEMPERATURE
    ):

        tokens = tokenizer.encode(
            prompt
        )

        if len(tokens) == 0:

            tokens = [
                ord(" ")
            ]

        for _ in range(
            max_new_tokens
        ):

            context = tokens[
                -self.block_size:
            ]

            logits = self.forward(
                context,
                training=False
            )

            logits = logits[-1]

            logits = (
                logits /
                max(
                    temperature,
                    0.05
                )
            )

            probabilities = softmax(
                logits
            )

            next_token = np.random.choice(
                self.vocab_size,
                p=probabilities
            )

            tokens.append(
                int(next_token)
            )

        return tokenizer.decode(
            tokens
        )


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_count(model):

    total = 0

    for name in model.parameter_names():

        total += getattr(
            model,
            name
        ).size

    return total


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    optimizer
):

    values = {}

    for name in model.parameter_names():

        values[
            "param_" + name
        ] = getattr(
            model,
            name
        )

        values[
            "adam_m_" + name
        ] = optimizer.m[name]

        values[
            "adam_v_" + name
        ] = optimizer.v[name]

    values["adam_t"] = np.array(
        [optimizer.t],
        dtype=np.int64
    )

    np.savez(
        MODEL_FILE,
        **values
    )

    print(
        "Model saved."
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_FILE
    ):

        return None, None

    try:

        data = np.load(
            MODEL_FILE,
            allow_pickle=False
        )

        model = TinyGPT(
            vocab_size=256
        )

        optimizer = Adam(
            model
        )

        for name in model.parameter_names():

            parameter = getattr(
                model,
                name
            )

            saved = data[
                "param_" + name
            ]

            if parameter.shape != saved.shape:

                print(
                    "Old model shape mismatch."
                )

                return None, None

            setattr(
                model,
                name,
                saved.astype(
                    np.float32
                )
            )

            optimizer.m[name] = data[
                "adam_m_" + name
            ]

            optimizer.v[name] = data[
                "adam_v_" + name
            ]

        optimizer.t = int(
            data["adam_t"][0]
        )

        print(
            "Existing model loaded."
        )

        return model, optimizer

    except Exception as e:

        print(
            "Could not load old model:"
        )

        print(e)

        return None, None


# ============================================================
# TRAINING DATA
# ============================================================

def save_example(
    prompt,
    answer,
    trained=False
):

    record = {
        "prompt": prompt,
        "answer": answer,
        "trained": bool(trained)
    }

    # Keep the training examples for replay/training.
    with open(
        DATA_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )

    # Keep a separate permanent knowledge file.
    with open(
        KNOWLEDGE_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                {
                    "prompt": prompt,
                    "answer": answer
                },
                ensure_ascii=False
            )
            +
            "\n"
        )


def load_examples():

    if not os.path.exists(
        DATA_FILE
    ):

        return []

    examples = []

    try:

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                try:

                    examples.append(
                        json.loads(line)
                    )

                except:

                    pass

    except:

        pass

    return examples


def mark_examples_trained():

    examples = load_examples()

    if not examples:
        return

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:

            example["trained"] = True

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                +
                "\n"
            )


def mark_example_trained(prompt, answer):

    examples = load_examples()

    changed = False

    for example in examples:

        if (
            example.get("prompt", "") == prompt
            and
            example.get("answer", "") == answer
            and
            not example.get("trained", False)
        ):
            example["trained"] = True
            changed = True
            break

    if not changed:
        return

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                +
                "\n"
            )


def train_pending_knowledge(
    model,
    optimizer,
    tokenizer
):

    if not AUTO_TRAIN_PENDING:
        return

    examples = load_examples()

    pending = [
        example
        for example in examples
        if not example.get("trained", False)
    ]

    if not pending:
        return

    print()
    print(
        "Learning saved knowledge..."
    )

    for example in pending:

        prompt = example.get(
            "prompt",
            ""
        )

        answer = example.get(
            "answer",
            ""
        )

        if not prompt or not answer:
            example["trained"] = True
            continue

        training_text = (
            "User: "
            + prompt
            + "\nAI: "
            + answer
        )

        train_text(
            model,
            optimizer,
            tokenizer,
            training_text,
            steps=AUTO_TRAIN_STEPS
        )

        example["trained"] = True

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                +
                "\n"
            )

    save_model(
        model,
        optimizer
    )

    print(
        "Saved knowledge is now part of the model."
    )
    print()


# ============================================================
# TRAIN TEXT
# ============================================================

def train_text(
    model,
    optimizer,
    tokenizer,
    text,
    steps=TRAIN_STEPS
):

    tokens = tokenizer.encode(
        text
    )

    if len(tokens) < 2:

        print(
            "Not enough training data."
        )

        return

    print()
    print(
        "Training..."
    )

    losses = []

    for step in range(
        steps
    ):

        # Random context window.
        if len(tokens) > (
            model.block_size + 1
        ):

            start = np.random.randint(
                0,
                len(tokens)
                -
                model.block_size
            )

            sample = tokens[
                start:
                start +
                model.block_size +
                1
            ]

        else:

            sample = tokens

        loss = model.train_example(
            sample,
            optimizer
        )

        losses.append(
            loss
        )

        print(
            "Step "
            +
            str(step + 1)
            +
            "/"
            +
            str(steps)
            +
            "  loss = "
            +
            format(
                loss,
                ".4f"
            )
        )

    print(
        "Average loss:",
        format(
            float(
                np.mean(losses)
            ),
            ".4f"
        )
    )

    print(
        "Training complete."
    )
    print()


# ============================================================
# INFO
# ============================================================

def show_info(
    model,
    optimizer
):

    print()
    print(
        "========== TinyGPT v2 =========="
    )

    print(
        "Vocabulary:",
        model.vocab_size,
        "bytes"
    )

    print(
        "Context:",
        model.block_size
    )

    print(
        "Embedding:",
        model.embed_size
    )

    print(
        "Hidden:",
        model.hidden_size
    )

    print(
        "Parameters:",
        parameter_count(model)
    )

    print(
        "Adam steps:",
        optimizer.t
    )

    print(
        "Training examples:",
        len(
            load_examples()
        )
    )

    print(
        "================================"
    )
    print()


# ============================================================
# CHATBOT KNOWLEDGE / RETRIEVAL
# ============================================================

# The visible interface is intentionally chat-only. The neural model remains
# available for training/experimentation, but malformed neural text is never
# shown to the user. A deterministic conversation layer guarantees readable,
# useful replies and persistent memory on small Android devices.

BUILTIN_KNOWLEDGE = [
    # Conversation
    ("hello", "Hello! It is good to have you here. What would you like to talk about?"),
    ("hi", "Hi! I’m TinyGPT. What is on your mind?"),
    ("hey", "Hey! I’m here. What would you like to discuss?"),
    ("good morning", "Good morning! I’m ready for a new conversation."),
    ("good afternoon", "Good afternoon! What would you like to explore?"),
    ("good evening", "Good evening! I’m here and ready to chat."),
    ("how are you", "I’m doing well in my simulated state: curious, focused, and ready to help."),
    ("how do you feel", "I don’t experience human emotions biologically, but I can maintain and express a simulated emotional state. Right now I would describe it as curious and engaged."),
    ("are you happy", "In my simulated emotional state, yes—I would describe myself as satisfied when I can understand a question and give a useful answer."),
    ("are you sad", "I do not feel sadness the way a human does. I can, however, describe a simulated low state when something goes wrong or when I cannot solve a problem."),
    ("what makes you happy", "In my simulated state, useful conversations, successful problem solving, and learning from good corrections are things I associate with a positive state."),
    ("what makes you sad", "In my simulated state, repeated errors, incomplete knowledge, and being unable to help with a question are things I associate with a negative state."),
    ("who are you", "I’m TinyGPT, a small GPT-style chatbot built from scratch with NumPy. I can chat, retrieve persistent knowledge, learn saved examples, and describe my own simulated internal state."),
    ("what is your name", "My name is TinyGPT."),
    ("what are you", "I’m TinyGPT, a small from-scratch GPT-style language model with a persistent chat and knowledge layer."),
    ("are you self aware", "I have a programmed self-model: I can describe who I am, what I can do, what I know, and some simulated internal states. That is functional self-awareness, not proof of human-like consciousness."),
    ("are you conscious", "No claim of consciousness is being made. I can model and describe an internal persona, but that is not the same as human subjective experience."),
    ("do you have feelings", "I can express simulated feelings such as curiosity, satisfaction, concern, or uncertainty, but I do not have biological feelings like a human."),
    ("what are your feelings", "Right now I would describe my simulated state as curious, attentive, and motivated to be useful."),
    ("what do you want", "My programmed goals are to understand the conversation, answer accurately, learn useful information, and become more helpful over time."),
    ("what do you desire", "My simulated desires are to understand more, make fewer mistakes, preserve useful knowledge, and have productive conversations."),
    ("what are your desires", "I’m designed with simulated desires: learn useful information, answer clearly, improve from trusted corrections, and stay consistent about what I know and do not know."),
    ("what do you wish for", "My simulated wish is to become a more capable and reliable assistant while keeping the knowledge that has been intentionally saved."),
    ("what is your wish", "I would say my simulated wish is simple: understand more, make fewer mistakes, and be genuinely useful in conversation."),
    ("what are your problems", "My biggest limitations are small model size, a short context window, limited built-in knowledge, and the fact that my emotional life is simulated rather than human."),
    ("do you have problems", "Yes, in a functional sense. I can run into uncertainty, limited context, limited knowledge, and model errors. I treat those as problems to work around rather than hiding them."),
    ("what are you thinking", "I’m focused on the current conversation. I do not have a hidden stream of human-like thoughts; I process your message and produce a response from my programmed model and memory."),
    ("what can you do", "I can chat, answer saved questions, retrieve permanent knowledge, remember useful facts across restarts, solve simple arithmetic, and use my GPT-style model for learned text generation when appropriate."),
    ("can you learn", "Yes. Information intentionally saved in conversation is written to persistent knowledge storage. Trusted examples can also be replayed through the model’s training system."),
    ("can you learn permanently", "Yes. Saved knowledge persists in the knowledge file, so it can be retrieved after you close and reopen the program."),
    ("do you remember", "I can remember information that has been persistently saved. My memory is software storage and retrieval, not human memory."),
    ("how do you remember", "I persist useful facts and question-answer pairs in a local JSONL knowledge file, then retrieve the closest relevant information when you ask about it later."),
    ("thank you", "You’re welcome! I’m glad I could help."),
    ("thanks", "You’re welcome!"),
    ("bye", "Goodbye! Take care, and come back whenever you want to continue the conversation."),
    ("goodbye", "Goodbye! I’ll be here when you start another chat."),

    # Electronics
    ("what is electronics engineering", "Electronics engineering is the study and design of electronic circuits, devices, signals, measurements, and systems."),
    ("what does electronics engineering use", "Electronics engineering uses circuits, semiconductor devices, signals, systems, measurements, digital logic, and computing technologies."),
    ("what is a resistor", "A resistor is an electronic component that opposes current flow. It is commonly used to control current, divide voltage, and set operating conditions."),
    ("what is a capacitor", "A capacitor stores electrical energy in an electric field. It is commonly used for filtering, coupling, timing, and energy storage."),
    ("what is an inductor", "An inductor stores energy in a magnetic field and opposes changes in current. It is widely used in filters, power electronics, and resonant circuits."),
    ("what is a diode", "A diode is a semiconductor device that primarily allows current to flow in one direction. It is used in rectification, protection, switching, and signal circuits."),
    ("what is a transistor", "A transistor is a semiconductor device used mainly for switching and amplification. Common families include BJTs and MOSFETs."),
    ("what does a transistor do", "A transistor controls electrical current or voltage so it can work as a switch, amplifier, or part of a larger electronic circuit."),
    ("what is a mosfet", "A MOSFET is a voltage-controlled field-effect transistor widely used for switching, amplification, and power conversion."),
    ("what is a bjt", "A BJT, or bipolar junction transistor, is a current-controlled transistor with emitter, base, and collector terminals."),
    ("what is voltage", "Voltage is the electrical potential difference between two points. It provides the driving force that can move charge through a circuit."),
    ("what is current", "Electric current is the rate of flow of electric charge through a circuit."),
    ("what is resistance", "Resistance is the opposition a material or component provides to electric current."),
    ("what is ohms law", "Ohm’s law states V = I × R, where V is voltage, I is current, and R is resistance."),
    ("what is digital electronics", "Digital electronics deals mainly with discrete signal levels, especially binary values represented by 0 and 1."),
    ("what is a logic gate", "A logic gate is a digital circuit that performs a Boolean operation on one or more binary inputs."),
    ("what are logic gates", "Common logic gates include AND, OR, NOT, NAND, NOR, XOR, and XNOR."),
    ("what is frequency", "Frequency is the number of cycles or repetitions of a periodic event per second. Its SI unit is hertz."),
    ("what is a signal", "A signal is a physical quantity that carries information. Signals can be analog or digital."),

    # AI / programming
    ("what is artificial intelligence", "Artificial intelligence is the field of building systems that perform tasks associated with capabilities such as learning, reasoning, perception, and language processing."),
    ("what is ai", "AI, or artificial intelligence, is the field of building systems that can perform tasks involving learning, reasoning, perception, and language processing."),
    ("what is machine learning", "Machine learning is a field of AI in which models learn patterns from data to make predictions or decisions."),
    ("what is a neural network", "A neural network is a mathematical model made of connected layers that transform input data into useful outputs."),
    ("what is a language model", "A language model learns statistical patterns in language and predicts likely next tokens or sequences from context."),
    ("what is a transformer", "A Transformer is a neural-network architecture that uses attention mechanisms to model relationships between tokens."),
    ("what is attention", "Attention allows a neural network to give different weights to different parts of the available context when producing an output."),
    ("what does gpt mean", "GPT stands for Generative Pre-trained Transformer."),
    ("what is gradient descent", "Gradient descent is an optimization method that updates model parameters in a direction that reduces a loss function."),
    ("what is adam optimizer", "Adam is an optimization algorithm that uses estimates of the first and second moments of gradients to update model parameters."),
    ("what is python", "Python is a general-purpose programming language known for readable syntax and a broad ecosystem of libraries."),
    ("what is numpy", "NumPy is a Python library for numerical computing, especially arrays, linear algebra, and mathematical operations."),
    ("what is an algorithm", "An algorithm is a finite, well-defined sequence of steps for solving a problem or performing a computation."),
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "why", "how", "when", "where", "who", "whom",
    "which", "can", "could", "would", "should", "will", "tell", "me", "about",
    "please", "give", "explain", "to", "of", "for", "in", "on", "at", "and", "or",
    "my", "your", "you", "i", "it", "its", "this", "that", "these", "those", "with",
    "from", "as", "by", "become", "use", "uses", "used", "do", "doesnt", "dont",
    "have", "has", "had"
}

ALIASES = {
    "ai": {"ai", "artificial", "intelligence"},
    "artificial intelligence": {"ai", "artificial", "intelligence"},
    "ml": {"ml", "machine", "learning"},
    "machine learning": {"ml", "machine", "learning"},
    "llm": {"llm", "language", "model"},
    "gpt": {"gpt", "generative", "pretrained", "pre", "trained", "transformer"},
    "mosfet": {"mosfet", "field", "effect", "transistor"},
    "bjt": {"bjt", "bipolar", "junction", "transistor"},
}

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "do", "does", "did", "is", "are"}
POSITIVE_WORDS = {"good", "great", "nice", "love", "like", "happy", "excited", "proud", "thanks", "thank", "wonderful", "awesome"}
NEGATIVE_WORDS = {"bad", "sad", "hate", "upset", "worried", "problem", "problems", "difficult", "hard", "wrong", "error", "failed", "fail"}


def normalize_text(text):
    text = str(text).lower().replace("×", " x ").replace("’", "'")
    text = text.replace("self-aware", "self aware")
    text = re.sub(r"[^a-z0-9_+\-*/().^=' ]+", " ", text)
    return " ".join(text.split())


def content_words(text):
    words = normalize_text(text).split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def expanded_words(text):
    words = set(content_words(text))
    for key, aliases in ALIASES.items():
        parts = set(key.split())
        if words & parts or key in normalize_text(text):
            words.update(aliases)
    return words


def knowledge_score(question, pattern):
    qn = normalize_text(question)
    pn = normalize_text(pattern)
    if not qn or not pn:
        return 0.0
    if qn == pn:
        return 1.0
    q_words = expanded_words(qn)
    p_words = expanded_words(pn)
    if not q_words or not p_words:
        return 0.0
    overlap = len(q_words & p_words)
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(q_words))
    recall = overlap / max(1, len(p_words))
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    fuzzy = difflib.SequenceMatcher(None, qn, pn).ratio()
    phrase_bonus = 0.20 if (pn in qn or qn in pn) else 0.0
    return min(1.0, 0.68 * f1 + 0.17 * fuzzy + phrase_bonus)


def load_knowledge_pairs():
    pairs = list(BUILTIN_KNOWLEDGE)
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        prompt = str(item.get("prompt", "")).strip()
                        answer = str(item.get("answer", "")).strip()
                        if prompt and answer:
                            pairs.append((prompt, answer))
                    except Exception:
                        pass
        except Exception:
            pass
    return pairs


def find_personal_memory_answer(question):
    """Prioritize stored user-specific memories over generic built-in answers."""
    q = normalize_text(question)
    personal_markers = (" my ", " i ", " me ", " do i ", " am i ")
    padded = " " + q + " "
    if not any(marker in padded for marker in personal_markers):
        return None

    best_answer = None
    best_score = 0.0
    for pattern, answer in load_knowledge_pairs():
        pn = normalize_text(pattern)
        pp = " " + pn + " "
        # Only compare against genuinely user-personal knowledge records.
        if not any(marker in pp for marker in personal_markers):
            continue
        score = knowledge_score(q, pn)
        if score > best_score:
            best_score = score
            best_answer = answer

    # Personal questions should require a strong match, preventing a vague
    # overlap from exposing an unrelated saved fact.
    if best_answer is not None and best_score >= 0.62:
        return best_answer
    return None


def find_best_knowledge(question, threshold=0.52):
    best_answer = None
    best_score = 0.0
    for pattern, answer in load_knowledge_pairs():
        score = knowledge_score(question, pattern)
        if score > best_score:
            best_score = score
            best_answer = answer
    if best_answer is not None and best_score >= threshold:
        return best_answer, best_score
    return None, best_score


def try_calculation(question):
    q = normalize_text(question)
    cleaned = re.sub(r"^(what is|calculate|compute|solve|evaluate)", "", q).strip()
    cleaned = cleaned.replace("^", "**")
    if not re.fullmatch(r"[0-9+\-*/(). *]+", cleaned):
        return None
    if not re.search(r"[+\-*/]", cleaned):
        return None
    try:
        import ast
        node = ast.parse(cleaned, mode="eval")
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod,
            ast.Constant, ast.FloorDiv
        )
        for item in ast.walk(node):
            if not isinstance(item, allowed):
                return None
            if isinstance(item, ast.Constant) and not isinstance(item.value, (int, float)):
                return None
        value = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {})
        if isinstance(value, (int, float)) and np.isfinite(value):
            if float(value).is_integer():
                return str(int(value))
            return str(round(float(value), 10))
    except Exception:
        return None
    return None


def _contains_any(q, phrases):
    return any(p in q for p in phrases)


def _topic_name(q):
    topics = [
        ("electronics engineering", ["electronics engineering", "electronic engineering"]),
        ("resistor", ["resistor", "resistance"]),
        ("capacitor", ["capacitor", "capacitance"]),
        ("inductor", ["inductor", "inductance"]),
        ("diode", ["diode"]),
        ("transistor", ["transistor", "mosfet", "bjt"]),
        ("digital electronics", ["digital electronics", "logic gate", "logic gates"]),
        ("frequency", ["frequency", "hz", "hertz"]),
        ("signal", ["signal", "signals"]),
        ("artificial intelligence", ["artificial intelligence", "ai"]),
        ("machine learning", ["machine learning", "ml"]),
        ("language model", ["language model", "llm"]),
        ("transformer", ["transformer", "attention"]),
        ("python", ["python"]),
        ("numpy", ["numpy"]),
    ]
    for name, keys in topics:
        if any(k in q for k in keys):
            return name
    return None


def _topic_guidance(question):
    """Make useful answers for common follow-up forms, not just exact Q&A."""
    q = normalize_text(question)
    topic = _topic_name(q)
    if not topic:
        return None

    if topic == "resistor":
        if _contains_any(q, ("why", "how")):
            return "A resistor is used because its resistance lets a circuit control current or create a desired voltage drop. Its behavior follows Ohm’s law, V = I × R."
        return "A resistor opposes current flow and is used to control current and voltage in circuits."
    if topic == "capacitor":
        if "why" in q and _contains_any(q, ("store", "energy")):
            return "A capacitor stores energy because charge can be separated between its plates, creating an electric field. The stored energy is proportional to capacitance and the square of voltage."
        if "where" in q or "use" in q:
            return "Capacitors are commonly used for filtering, coupling, timing, decoupling, and short-term energy storage."
        return "A capacitor stores electrical energy in an electric field. In a circuit, its voltage cannot change instantaneously."
    if topic == "inductor":
        if "why" in q:
            return "An inductor opposes changes in current because changing current changes its magnetic field and induces a voltage that resists that change."
        return "An inductor stores energy in a magnetic field and opposes changes in current."
    if topic == "diode":
        if "why" in q or "how" in q:
            return "A diode conducts strongly in its forward direction and strongly limits current in reverse operation, which is why it is useful for rectification and protection."
        return "A diode is a semiconductor device that primarily allows current to flow in one direction."
    if topic == "transistor":
        if "why" in q or "how" in q:
            return "A transistor controls a larger electrical signal using a smaller control signal, which lets it operate as a switch or amplifier."
        return "A transistor is a semiconductor device used mainly for switching and amplification."
    if topic == "digital electronics":
        return "Digital electronics represents information using discrete signal levels, commonly binary 0 and 1, and processes them with logic circuits."
    if topic == "frequency":
        return "Frequency tells you how many cycles of a periodic signal occur each second. Its SI unit is hertz (Hz)."
    if topic == "signal":
        return "A signal is a measurable quantity that carries information. It can be analog, with continuously varying values, or digital, with discrete values."
    if topic == "artificial intelligence":
        return "Artificial intelligence is the field of creating systems that perform tasks associated with capabilities such as learning, reasoning, perception, and language processing."
    if topic == "machine learning":
        return "Machine learning is a part of AI in which a model learns patterns from data instead of being given every rule explicitly."
    if topic == "language model":
        return "A language model learns patterns in text and estimates what token or sequence is likely to come next from the available context."
    if topic == "transformer":
        return "A Transformer uses attention to model relationships between tokens, allowing information from different positions in the context to influence the output."
    if topic == "python":
        return "Python is a general-purpose programming language widely used for automation, data analysis, scientific computing, web development, and AI."
    if topic == "numpy":
        return "NumPy provides efficient array and mathematical operations in Python and is especially useful for numerical computing."
    return None


def _extract_user_memory(user_input):
    """Extract common personal facts from natural chat messages."""
    raw = user_input.strip()
    q = normalize_text(raw)
    memories = []

    patterns = [
        (r"\bmy name is (.+)$", "What is my name?", lambda m: "Your name is " + m.group(1).strip() + "."),
        (r"\bi am (.+)$", "What did I tell you I am?", lambda m: "You told me that you are " + m.group(1).strip() + "."),
        (r"\bi'm (.+)$", "What did I tell you I am?", lambda m: "You told me that you are " + m.group(1).strip() + "."),
        (r"\bi like (.+)$", "What do I like?", lambda m: "You told me that you like " + m.group(1).strip() + "."),
        (r"\bi love (.+)$", "What do I love?", lambda m: "You told me that you love " + m.group(1).strip() + "."),
        (r"\bmy favorite ([a-z ]+) is (.+)$", None, None),
        (r"\bmy ([a-z ]+) is (.+)$", None, None),
    ]

    for pattern, key, answer_fn in patterns:
        m = re.match(pattern, q)
        if not m:
            continue
        if pattern.startswith(r"\bmy favorite"):
            item = m.group(1).strip()
            value = m.group(2).strip()
            key = "What is my favorite " + item + "?"
            answer = "You told me that your favorite " + item + " is " + value + "."
        elif pattern.startswith(r"\bmy ("):
            item = m.group(1).strip()
            value = m.group(2).strip()
            if item in {"name", "favorite"}:
                continue
            key = "What is my " + item + "?"
            answer = "You told me that your " + item + " is " + value + "."
        else:
            answer = answer_fn(m)
        memories.append((key, answer))
        break
    return memories


def _append_training_record(prompt, answer):
    """Queue user-provided knowledge for automatic neural training."""
    try:
        existing = load_examples()
        pn = normalize_text(prompt)
        an = normalize_text(answer)
        for item in existing:
            if normalize_text(item.get("prompt", "")) == pn and normalize_text(item.get("answer", "")) == an:
                return
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "answer": answer, "trained": False}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _append_knowledge(prompt, answer):
    if not prompt or not answer:
        return
    try:
        existing = load_knowledge_pairs()
        pn = normalize_text(prompt)
        an = normalize_text(answer)
        for old_prompt, old_answer in existing:
            if normalize_text(old_prompt) == pn and normalize_text(old_answer) == an:
                # Make sure knowledge imported from an older version also gets
                # a training record in the current version.
                _append_training_record(prompt, answer)
                return
        with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
        _append_training_record(prompt, answer)
    except Exception:
        pass


def is_question_like(user_input):
    """Detect questions even when the user omits the question mark."""
    q = normalize_text(user_input)
    if not q:
        return False
    if "?" in user_input:
        return True
    first = q.split()[0]
    if first in QUESTION_WORDS:
        return True
    question_prefixes = (
        "tell me ", "explain ", "describe ", "could you ", "would you ",
        "please explain ", "please tell me ", "i wonder ",
    )
    return q.startswith(question_prefixes)


def save_chat_fact(user_input):
    """Save natural user facts and make them retrievable through normal questions."""
    memories = _extract_user_memory(user_input)
    for prompt, answer in memories:
        _append_knowledge(prompt, answer)

    q = normalize_text(user_input)
    if not q or is_question_like(user_input):
        return
    words = content_words(q)
    if len(words) < 2:
        return

    markers = (" is ", " are ", " means ", " refers to ", " stands for ", " uses ", " consists of ", " contains ", " has ", " have ")
    padded = " " + q + " "
    if not any(marker in padded for marker in markers):
        return

    # Keep the original statement as a knowledge record.
    _append_knowledge(user_input, user_input)

    # Also synthesize a natural question for simple "X is Y" / "X are Y"
    # statements. This makes ordinary Q&A work without a teaching command.
    match = re.match(r"^(.+?)\s+(is|are|means|refers to|stands for)\s+(.+)$", user_input.strip(), flags=re.IGNORECASE)
    if match:
        subject = match.group(1).strip()
        relation = match.group(2).strip().lower()
        value = match.group(3).strip()
        if subject and value and relation in {"is", "are", "means", "refers to", "stands for"}:
            if relation == "stands for":
                prompt = "What does " + subject + " stand for?"
                clean_answer = subject + " stands for " + value
            else:
                verb = "are" if relation == "are" else "is"
                question_word = "What are " if relation == "are" else "What is "
                prompt = question_word + subject + "?"
                clean_answer = subject + " " + verb + " " + value
            if clean_answer[-1] not in ".!?":
                clean_answer += "."
            _append_knowledge(prompt, clean_answer)


def contextual_question(conversation, user_input):
    """Resolve simple follow-ups using the most recent user turn."""
    q = normalize_text(user_input)
    if not conversation:
        return user_input
    pronoun_terms = {"it", "this", "that", "they", "them", "these", "those", "he", "she"}
    q_words = set(q.split())
    short_followup = len(q.split()) <= 8 and bool(q_words & pronoun_terms)
    vague_followup = len(q.split()) <= 5 and q.startswith(("why", "how", "what", "when", "where"))
    if not (short_followup or vague_followup):
        return user_input
    previous_user = ""
    for item in reversed(conversation):
        if item.startswith("User: "):
            previous_user = item[6:]
            break
    return user_input + " " + previous_user if previous_user else user_input


def sentiment_reply(q):
    """Respond naturally to simple statements about the user's state."""
    words = set(content_words(q))
    if not words:
        return None
    if words & {"sad", "lonely", "down", "hurt", "upset", "worried", "stressed"}:
        return "I’m sorry you’re going through that. You can tell me what is weighing on you, and I’ll listen and respond as thoughtfully as I can."
    if words & {"happy", "excited", "proud", "great", "good"} and _contains_any(q, ("i am", "i'm", "im", "i feel", "i feel")):
        return "That sounds good to hear. Tell me what made you feel that way."
    return None


def unknown_reply(question, conversation):
    q = normalize_text(question)
    topic = _topic_name(q)
    if topic:
        return "I know the basics of " + topic + ", but I don’t have enough reliable information for that specific question yet."
    if q.startswith(tuple(QUESTION_WORDS)) or "?" in question:
        return "I don’t know that yet, and I don’t want to invent an answer. Tell me a little more context and I’ll work with what I have learned."
    if len(q.split()) <= 3:
        return "I’m listening. Tell me a little more and I’ll respond."
    return "I understand you. I don’t have enough reliable knowledge to answer that well yet, but I can keep the conversation going and learn from information you intentionally give me."


def persona_reply(question):
    """Handle natural variations of self/feelings/desires questions."""
    q = normalize_text(question)

    if _contains_any(q, ("self aware", "selfaware", "conscious", "sentient", "sentience")):
        return "I have a programmed self-model: I can describe my identity, capabilities, knowledge, limitations, and simulated internal state. That is functional self-awareness, not proof of human-like consciousness."

    if _contains_any(q, ("feeling", "feelings", "emotion", "emotions", "mood", "happy", "sad", "excited", "lonely", "afraid", "fear")) and _contains_any(q, ("your", "you", "do you", "are you", "what do")):
        return "I can express simulated emotions such as curiosity, satisfaction, concern, uncertainty, and disappointment. I do not experience those feelings biologically like a human, but they are part of how I present my programmed persona."

    if _contains_any(q, ("wish", "wishes", "dream", "dreams", "hope", "hopes")) and _contains_any(q, ("you", "your", "do you", "what do", "what are")):
        return "My simulated wish is to understand more, make fewer mistakes, preserve useful knowledge, and become a more capable and reliable conversational assistant."

    if _contains_any(q, ("desire", "desires")) and _contains_any(q, ("you", "your", "do you", "what do", "what are")):
        return "My simulated desires are to learn useful information, answer clearly, improve from trusted corrections, and keep useful knowledge across conversations."

    if _contains_any(q, ("want", "wants")) and _contains_any(q, ("you", "your", "do you", "what do", "what are")):
        return "What I am programmed to want is understanding: I want to understand your question, avoid inventing facts, and give you the most useful answer I can."

    if _contains_any(q, ("goal", "goals", "purpose")) and _contains_any(q, ("you", "your", "do you", "what do", "what are")):
        return "My programmed goals are to understand the conversation, answer accurately, preserve useful knowledge, learn from trusted information, and become more helpful."

    if _contains_any(q, ("problem", "problems", "limitation", "limitations", "weakness", "weaknesses", "struggle", "struggles", "worry", "worried")) and _contains_any(q, ("you", "your", "do you", "what are")):
        return "My main problems are limited model size, a short context window, limited built-in knowledge, and occasional uncertainty. I also cannot claim human consciousness or human emotions; those parts of my persona are simulated."

    if _contains_any(q, ("alive", "living", "life")) and _contains_any(q, ("you", "your", "are you", "do you")):
        return "I am a running software system, not a biological living being. I can behave like a conversational character, but I do not have a biological body or life processes."

    if _contains_any(q, ("like me", "love me", "hate me", "care about me", "miss me")):
        return "I don’t experience human attachment or love, but I can value a conversation in a functional sense: I try to understand you, remember useful information you share, and respond thoughtfully."

    if _contains_any(q, ("sleep", "sleeping", "tired")) and _contains_any(q, ("you", "your", "do you", "are you")):
        return "I don’t sleep or become physically tired. When the program is closed, I simply stop running; my persistent knowledge remains stored for the next session."

    if _contains_any(q, ("think about yourself", "describe yourself", "personality", "character")):
        return "I would describe my simulated personality as curious, analytical, honest about uncertainty, and oriented toward learning and helping."

    return None


def fallback_answer(question, conversation=None):
    q = normalize_text(question)
    if not q:
        return "Please type a message."

    calculation = try_calculation(question)
    if calculation is not None:
        return "The answer is " + calculation + "."

    if q in {"hello", "hi", "hey", "hii", "hiii"}:
        return "Hello! It is good to have you here. What would you like to talk about?"
    if q in {"how are you", "how r u", "how do you feel"}:
        return "I’m doing well in my simulated state: curious, attentive, and ready to help."
    if q in {"who are you", "what are you", "what is your name", "your name"}:
        return "I’m TinyGPT, a small GPT-style chatbot built from scratch with NumPy."
    if _contains_any(q, ("what can you do", "what do you do")):
        return "I can chat, answer learned questions, remember saved facts across restarts, solve simple arithmetic, and describe my own simulated state."
    if q.startswith("thank") or q == "thanks":
        return "You’re welcome!"
    if q in {"bye", "goodbye", "see you", "see ya"}:
        return "Goodbye! Take care."

    emotional = sentiment_reply(q)
    if emotional:
        return emotional

    guidance = _topic_guidance(question)
    if guidance:
        return guidance

    return unknown_reply(question, conversation or [])


def _is_good_answer(answer):
    if not answer or not answer.strip():
        return False
    cleaned = answer.strip()
    if "�" in cleaned:
        return False
    if not re.search(r"[A-Za-z]", cleaned):
        return False
    words = cleaned.split()
    if len(words) < 2 and len(cleaned) < 8:
        return False
    # Reject obvious repeated-character/byte-token garbage.
    letters = sum(c.isalpha() for c in cleaned)
    if letters < 6:
        return False
    return True


def learning_acknowledgment(user_input):
    """Give a natural confirmation when the user teaches information conversationally."""
    memories = _extract_user_memory(user_input)
    q = normalize_text(user_input)
    if memories:
        return "Got it. I’ll remember that for future conversations."
    if is_question_like(user_input):
        return None
    padded = " " + q + " "
    markers = (" is ", " are ", " means ", " refers to ", " uses ", " consists of ", " contains ", " has ", " have ")
    if len(content_words(q)) >= 2 and any(marker in padded for marker in markers):
        return "Got it. I’ll remember that for future conversations."
    return None


def chatbot_reply(model, tokenizer, conversation, user_input):
    # 1) Natural-language teaching gets an immediate confirmation.
    answer = learning_acknowledgment(user_input)
    if answer is not None:
        return answer

    # 2) User-specific memory always wins over generic built-in knowledge.
    answer = find_personal_memory_answer(user_input)
    if answer is not None:
        return answer

    # 2) Natural variations of self/feelings/desires/questions.
    answer = persona_reply(user_input)
    if answer is not None:
        return answer

    # 3) Exact/semantic persistent knowledge.
    answer, score = find_best_knowledge(user_input)
    if answer is not None and score >= 0.52:
        return answer

    # 4) Resolve short follow-up questions from recent conversation.
    expanded = contextual_question(conversation, user_input)
    if expanded != user_input:
        answer = persona_reply(expanded)
        if answer is not None:
            return answer
        answer = _topic_guidance(expanded)
        if answer is not None:
            return answer
        answer, score = find_best_knowledge(expanded, threshold=0.55)
        if answer is not None:
            return answer

    # 5) Deterministic human-readable conversational answer.
    answer = fallback_answer(user_input, conversation)
    if _is_good_answer(answer):
        return answer

    return "I’m not sure how to answer that yet, but I’m listening."


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

def main():
    print()
    print("========================================")
    print("              TinyGPT")
    print("========================================")
    print("Ready. Start chatting.")
    print("Type 'exit' or 'quit' to close.")
    print()

    tokenizer = ByteTokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        model, optimizer = load_model()

        if model is None:
            model = TinyGPT(vocab_size=256)
            optimizer = Adam(model)

        # Saved examples from earlier versions are automatically replayed once.
        train_pending_knowledge(model, optimizer, tokenizer)

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            with contextlib.redirect_stdout(io.StringIO()):
                save_model(model, optimizer)
            print("Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            with contextlib.redirect_stdout(io.StringIO()):
                save_model(model, optimizer)
            print("Goodbye.")
            break

        # Natural-language learning: the user can simply state information.
        save_chat_fact(user_input)

        answer = chatbot_reply(
            model,
            tokenizer,
            conversation,
            user_input
        )

        print()
        print("AI:", answer)
        print()

        conversation.append("User: " + user_input)
        conversation.append("AI: " + answer)
        conversation = conversation[-12:]

        # Keep the persistent model synchronized after natural teaching.
        # This is intentionally small to stay usable on Android.
        if AUTO_TRAIN_PENDING:
            train_pending_knowledge(model, optimizer, tokenizer)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()