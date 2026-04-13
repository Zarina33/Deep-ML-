"""
GPT-2 Text Generation
Hard
NLP


Implement a Simplified GPT-2-like Text Generation Function
You are tasked with implementing a simplified GPT-2-like text generation function in Python. This function will incorporate the following components of a minimal GPT-2 architecture:
Token Embeddings: Map input tokens to dense vector representations.
Positional Embeddings: Add positional information to token embeddings.
Multi-head Attention: Attend to various parts of the sequence.
Feed-Forward Network: Process attention outputs through a dense layer.
Layer Normalization: Stabilize the training process.
The function must take in the following parameters:
Prompt: The initial text to guide the generation process.
Number of Tokens to Generate: Specify how many tokens to output.
Your function should output the generated text.
Additionally, utilize the helper function load_encoder_hparams_and_params to retrieve:
A dummy encoder.
Model hyperparameters.
Model parameters.
Build your text generation logic around these components. This exercise is designed to help you understand the core concepts behind GPT-2's autoregressive text generation.
Example:
Input:
prompt="hello", n_tokens_to_generate=5
Output:
hello hello hello <UNK> <UNK>
Reasoning:
The function encodes the input "hello" into tokens using the dummy encoder, then runs a simplified GPT-2 forward pass to generate 5 tokens. Finally, it decodes the generated tokens back into text.
"""

import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b  # ✅ return на том же уровне

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
#                          ↑ np.   не n.                   ↑ 0.044715 не 0.44715

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)  # ✅ добавили return =

def ffn(x, mlp):
    a = x @ mlp["c_fc"]["w"] + mlp["c_fc"]["b"]
    a = gelu(a)
    return a @ mlp["c_proj"]["w"] + mlp["c_proj"]["b"]  # ✅ return на том же уровне

def attention(x, attn, n_head):
    qkv = x @ attn["c_attn"]["w"] + attn["c_attn"]["b"]
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(m):  # ✅ внутри attention с отступом
        return m.reshape(m.shape[0], n_head, -1).transpose(1, 0, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scale = np.sqrt(q.shape[-1])
    scores = q @ k.transpose(0, 2, 1) / scale
    n = x.shape[0]  # ✅ n = не n.
    mask = (1 - np.tri(n)) * -1e10
    scores += mask
    out = softmax(scores) @ v
    out = out.transpose(1, 0, 2).reshape(n, -1)
    return out @ attn["c_proj"]["w"] + attn["c_proj"]["b"]  # ✅ return на том же уровне

def transformer(x, block, n_head):
    x = x + attention(layer_norm(x, **block["ln_1"]), block["attn"], n_head)
    x = x + ffn(layer_norm(x, **block["ln_2"]), block["mlp"])
    return x  # ✅ return на том же уровне

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    encoder, hparams, params = load_encoder_hparams_and_params()
    inputs = encoder.encode(prompt)
    original_inputs = len(inputs)  # ✅ запоминаем длину до генерации

    for _ in range(n_tokens_to_generate):
        x = params["wte"][inputs] + params["wpe"][range(len(inputs))]
        for block in params["blocks"]:
            x = transformer(x, block, hparams["n_head"])
        x = layer_norm(x, **params["ln_f"])
        logits = x[-1] @ params["wte"].T
        next_token = int(np.argmax(logits))
        inputs.append(next_token)

    return encoder.decode(inputs[original_inputs:])  # ✅ только новые токены

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

    hparams = {"n_ctx": 1024, "n_head": 2}

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [{
            "mlp": {
                "c_fc":   {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
                "c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
            },
            "attn": {
                "c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
                "c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
            },
            "ln_1": {"g": np.ones(10), "b": np.zeros(10)},
            "ln_2": {"g": np.ones(10), "b": np.zeros(10)},
        }],
        "ln_f": {"g": np.ones(10), "b": np.zeros(10)}
    }

    encoder = DummyBPE()
    return encoder, hparams, params