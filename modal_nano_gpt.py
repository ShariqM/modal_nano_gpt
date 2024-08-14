import modal
from modal import Image, build, enter, method
from dataclasses import dataclass


##################
### Image Code ###
##################
# gpu = None
gpu = "A10G"
# gpu = "T4"
def image_setup():
    import requests
    import os

    # Download dataset if not there yet.
    input_file_path = os.path.join(os.path.dirname(__file__), 'shakespeare_char.txt')
    if not os.path.exists(input_file_path):
        print ("Downloading Shakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    else:
        print ("Shakespeare dataset already here.")


app = modal.App("modal_nano_gpt")
torch_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "requests==2.26.0",
    )
    .run_function(image_setup)
)

with torch_image.imports():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from timeit import default_timer as timer

###################
### Remote Code ###
###################
@app.cls(image=torch_image, gpu=gpu)
class Dataset(object):
    """Manage text dataset and encoding & decoding."""

    def __init__(self, txt_filename, train_percent, batch_size,
                 context_size, n_eval_steps, device):
        with open(txt_filename, 'r', encoding='utf-8') as f:
            text = f.read()
        self.device = device
        self.batch_size = batch_size
        self.context_size = context_size
        self.n_eval_steps = n_eval_steps
        assert (train_percent > 0.0) and (train_percent < 1.0), (
            "train_percent must be in (0,1)")


        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print (f"Vocab Size: {self.vocab_size}")
        print ('Unique letters: ', ''.join(chars))

        stoi = {c:i for i,c in enumerate(chars)}
        itos = {i:c for i,c in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: [itos[i] for i in l]
        self.newline_encoded = self.encode(['\n'])[0]

        # Train/Validation.
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = len(data)
        self.train_data = data[:int(train_percent*n)]
        self.val_data = data[int(train_percent*n):]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data

        starts = torch.randint(
            len(data) - self.context_size, (self.batch_size,))
        x = torch.stack(
            [data[start:start+self.context_size] for start in starts])
        y = torch.stack(
            [data[start+1:start+self.context_size+1] for start in starts])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def eval_model(self, model):
        out = {}
        model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(self.n_eval_steps)
            for k in range(self.n_eval_steps):
                xb, yb = self.get_batch(split)
                logits, loss = model.forward(xb, yb) # Modal: Why need forward?
                losses[k] = loss
            out[split] = losses.mean()
        model.train()
        return out

#######################
### Attention Model ###
#######################
class Head(nn.Module):
    """One head self-attention."""

    def __init__(self, hparams, input_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(input_size, head_size, bias=False)
        self.key = nn.Linear(input_size, head_size, bias=False)
        self.value = nn.Linear(input_size, head_size, bias=False)
        self.dropout = nn.Dropout(hparams.dropout)

        self.register_buffer('tril',
            torch.tril(
                torch.ones(hparams.context_size, hparams.context_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)  # bth
        k = self.key(x)  # bth
        v = self.value(x)  # bth

        weight = torch.einsum("bth,buh->btu", q, k)
        # Causal Mask
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        # Normalize to reduce output varirance to 1
        weight /= np.sqrt(self.head_size)
        dist = F.softmax(weight, dim=-1)  # btt
        dist = self.dropout(dist)

        out = torch.einsum("btu,buh->bth", dist, v)  # bth

        return out

class MultiHeadFast(nn.Module):
    """Multihead self-attention."""

    def __init__(self, hparams, input_size):
        super().__init__()
        self.input_size = input_size
        self.head_size = input_size // hparams.n_heads
        self.n_heads = hparams.n_heads
        self.dropout = hparams.dropout

        # Parallel Head calculation
        self.qkv_proj = nn.Linear(input_size, 3 * input_size, bias=False)
        self.use_flash_attention = (
            hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        self.register_buffer('tril',
            torch.tril(
                torch.ones(hparams.context_size, hparams.context_size)
                .view(1, 1, hparams.context_size, hparams.context_size)))
        self.head_dropout = nn.Dropout(hparams.dropout)

        # Multi Head operaitons
        self.proj = nn.Linear(input_size, input_size)
        self.out_dropout = nn.Dropout(hparams.dropout)

    def forward(self, x):
        B, T, C = x.shape

        # QKV for all heads
        qkv = self.qkv_proj(x) # bt(3i)
        q, k, v = qkv.split(self.input_size, dim=-1)

        # Split heads
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2) # bnth
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2) # bnth
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2) # bnth

        if self.use_flash_attention:
            heads_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout,
                is_causal=True)
        else:
            weight = torch.einsum("bnth,bnuh->bntu", q, k)
            weight /= np.sqrt(self.head_size)
            weight = weight.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
            dist = F.softmax(weight, dim=-1)
            dist = self.head_dropout(dist)

            heads_out = torch.einsum("bntu,bnuh->bnth", dist, v)

        multi_head_out = heads_out.transpose(1, 2).reshape(B, T, C) # bth
        return self.out_dropout(self.proj(multi_head_out))

class MultiHead(nn.Module):
    """Multihead self-attention."""

    def __init__(self, hparams, input_size):
        super().__init__()
        head_size = input_size // hparams.n_heads
        self.heads = nn.ModuleList([Head(hparams, input_size, head_size)
            for _ in range(hparams.n_heads)])
        self.proj = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x):
        sa_out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(sa_out))

class MLP(nn.Module):
    """ One layer MLP."""

    def __init__(self, hparams, input_size):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size, 4 * input_size),
                nn.ReLU(),
                nn.Linear(4 * input_size, input_size),
                nn.Dropout(hparams.dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # self.sa_heads = MultiHead(hparams, hparams.n_embed)
        self.sa_heads = MultiHeadFast(hparams, hparams.n_embed)
        self.mlp = MLP(hparams, hparams.n_embed)
        self.ln1 = nn.LayerNorm(hparams.n_embed)
        self.ln2 = nn.LayerNorm(hparams.n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@app.cls(image=torch_image, gpu=gpu)
class AttentionModel(nn.Module):
    def __init__(self, vocab_size, hparams, device):
        super().__init__()
        self.context_size = hparams.context_size
        self.device = device

        self.token_embedding_table = nn.Embedding(
                vocab_size, hparams.n_embed, device=device)
        self.pos_embedding_table = nn.Embedding(
                hparams.context_size, hparams.n_embed)
        self.blocks = nn.Sequential(
            *[Block(hparams) for _ in range(hparams.n_blocks)])

        self.ln_f = nn.LayerNorm(hparams.n_embed)
        self.lm_head = nn.Linear(hparams.n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx - (B, T)
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.pos_embedding_table(
                torch.arange(T, device=self.device))
        embedding = token_embedding + position_embedding
        x = self.blocks(embedding)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            xlogits = logits.view(logits.shape[0] * logits.shape[1], -1)
            xtargets = targets.view(-1)
            loss = F.cross_entropy(xlogits, xtargets)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            # Make predictions
            # pdb.set_trace()
            logits = self(idx[:, -self.context_size:])[0] # B,T,C
            logits = logits[:,-1,:] # B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], axis=1)
        return idx


@app.cls(image=torch_image, gpu=gpu)
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx - (B, T)
        logits = self.token_embedding_table(idx)

        if targets is not None:
            xlogits = logits.view(logits.shape[0] * logits.shape[1], -1)
            xtargets = targets.view(-1)
            loss = F.cross_entropy(xlogits, xtargets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Make predictions
            logits = self(idx)[0] # B,T,C
            logits = logits[:,-1,:] # B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], axis=1)
        return idx

@dataclass
class ModelHyperparameters:
    # Fast
    # n_heads: int = 4
    # n_embed: int = 32
    # n_blocks: int = 3
    # context_size: int = 16
    # dropout: float = 0.2

    # Mid
    # n_heads: int = 4
    # n_embed: int = 128
    # n_blocks: int = 4
    # context_size: int = 64
    # dropout: float = 0.2

    # Karpathy
    n_heads: int = 6
    n_embed: int = 384
    n_blocks: int = 6
    context_size: int = 256
    dropout: float = 0.2

@app.function(image=torch_image, gpu=gpu)
def func():
    #######################
    ### Hyperparameters ###
    #######################
    # Training
    batch_size = 64
    n_steps = 5000
    n_eval_steps = 100
    n_steps_before_eval = int(n_steps/10.) # eval every 10% of training
    train_percent = 0.9
    # learning_rate = 1e-2
    # learning_rate = 1e-3
    learning_rate = 3e-4

    # Model params
    hparams = ModelHyperparameters()
    assert (hparams.n_embed % hparams.n_heads == 0), (
            "n_embed must be divisible by n_heads")
    # Misc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print (f'Remote Device: {device} // {gpu}')

    #########################
    ### Data & Model prep ###
    #########################
    # Construct dataset
    dataset = Dataset('shakespeare_char.txt', train_percent,
                      batch_size, hparams.context_size, n_eval_steps, device)

    # Build Model
    # model = BigramModel(dataset.vocab_size)
    model = AttentionModel(dataset.vocab_size, hparams, device)
    m = model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print (f"Num parameters: {num_parameters}")

    # Helper function for kicking off model generation
    def generate(model, n_new_tokens):
        gen_start = (
            dataset.newline_encoded * torch.ones((1, 1), dtype=torch.long))
        gen_start = gen_start.to(device)
        gen_out = model.generate(gen_start, n_new_tokens)[0] # 0th batch
        chars_out = dataset.decode([x for x in gen_out.tolist()])
        str_out = "".join(chars_out)
        return str_out

    n_new_tokens = 100
    print ("Before Training Generation: ", generate(model, n_new_tokens))

    ################
    ### Training ###
    ################
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    t_last = timer()
    for step in range(n_steps):
        # sample
        xb, yb = dataset.get_batch('train')

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % n_steps_before_eval == 0:
            out = dataset.eval_model(model)
            runtime_s = timer() - t_last
            print (f"{step:5d}) // {runtime_s:>5.2f}s"
                f" // Train Loss: {out['train']:.2f} // Val Loss:"
                f" {out['val']:.2f}")
            t_last = timer()

    out = dataset.eval_model(model)
    runtime_s = timer() - t_last
    print (f"Final) // {runtime_s:>5.2f}"
        f" // Train Loss: {out['train']:.2f} // Val Loss:"
        f" {out['val']:.2f}")
    print (f"Num parameters: {num_parameters}")

    n_new_tokens = 1000
    print ("After Training Generation: ", generate(model, n_new_tokens))
    return -1


@app.local_entrypoint()
def main():
    ret = func.remote()
    print ('returned: ', ret)
    print ('not run on modal')
