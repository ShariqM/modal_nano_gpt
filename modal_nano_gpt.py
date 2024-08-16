import modal
from modal import Image, build, enter, method
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse

# import argparse
# parser = argparse.ArgumentParser(description="Pass args to monitor")
# parser.add_argument(
    # "-c",
    # type=str,
    # dest="monitor_experiment_name",
    # default=None,
    # required=False,
# )
# FLAGS = parser.parse_args()


##############
### Setup ####
##############
# size = 'tiny'   # CPU
size = 'small'  # CPU/GPU
# size = 'medium' # GPU

# gpu = None
gpu = "T4"
# gpu = "A10G"

timeout = 20 * 60  # 20 minutes
# timeout =  5 * 60  # Default 5 minutes

@dataclass
class ModelHyperparameters:
    if size == 'tiny':
        n_heads: int = 4
        n_embed: int = 32
        n_blocks: int = 3
        context_size: int = 16
        dropout: float = 0.2
    elif size == 'small':
        n_heads: int = 4
        n_embed: int = 128
        n_blocks: int = 4
        context_size: int = 64
        dropout: float = 0.2
    else:
        # https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5976s
        n_heads: int = 6
        n_embed: int = 384
        n_blocks: int = 6
        context_size: int = 256
        dropout: float = 0.2


volume = modal.Volume.from_name("nano_gpt_volume")
volume_path = Path("/vol/data")
model_filename = "nano_gpt_model.pt"
# model_filename = "nano_gpt_model_v0.2.pt"
medium_model_path = volume_path / "medium" / model_filename
log_path = volume_path / "logs"
save_path = volume_path / "models"

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"

app = modal.App("modal_nano_gpt")
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "gradio~=3.50.2",
        "tensorboard",
    )
)

with image.imports():
    import numpy as np
    import os
    import glob
    from timeit import default_timer as timer

    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    import tensorboard
    from torch.utils.tensorboard import SummaryWriter

def print_banner(string):
    print ('#' * (len(string) + 8))
    print (f'### {string} ###')
    print ('#' * (len(string) + 8))

###############
### Dataset ###
###############
@app.cls(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=timeout)
class Dataset(object):
    """Manage text dataset and encoding & decoding."""

    def __init__(self, txt_filename, train_percent, batch_size,
                 context_size, n_eval_steps, device):
        with open(volume_path / txt_filename, 'r', encoding='utf-8') as f:
            text = f.read()
        self.device = device
        self.batch_size = batch_size
        self.context_size = context_size
        self.n_eval_steps = n_eval_steps
        assert (train_percent > 0.0) and (train_percent < 1.0), (
            "train_percent must be in (0,1)")

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        print (f"Vocab Size: {self.vocab_size}")
        print ('Unique letters: ', ''.join(self.chars))

        stoi = {c:i for i,c in enumerate(self.chars)}
        itos = {i:c for i,c in enumerate(self.chars)}
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

class MLP(nn.Module):
    """ Multi-Layer Perception (last ops of each block)."""

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
        self.sa_heads = MultiHeadFast(hparams, hparams.n_embed)
        self.mlp = MLP(hparams, hparams.n_embed)
        self.ln1 = nn.LayerNorm(hparams.n_embed)
        self.ln2 = nn.LayerNorm(hparams.n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@app.cls(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=timeout)
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
            logits = self(idx[:, -self.context_size:])[0] # B,T,C
            logits = logits[:,-1,:] # B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], axis=1)
        return idx

@app.function(
    image=image, # Without this, dataset crashes with "torch not found"
    volumes={volume_path: volume})
@modal.wsgi_app()
def monitor_training():
    # if FLAGS.monitor_experiment_name is not None:
        # monitor_path = log_path / FLAGS.monitor_experiment_name
    # else:
        # log_paths = glob.glob(f"{log_path}/*")
        # latest_log_path = max(log_paths, key=os.path.getctime)
        # monitor_path = Path(latest_log_path)
    log_paths = glob.glob(f"{log_path}/*")
    latest_log_path = max(log_paths, key=os.path.getctime)
    monitor_path = Path(latest_log_path)
    print_banner(f"Monitoring: {monitor_path.name}")

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(monitor_path))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app


@app.function(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=timeout)
def train_model(hparams, experiment_name, run_to_first_save=False):
    #######################
    ### Hyperparameters ###
    #######################
    # Training
    batch_size = 64
    n_steps = 5000
    n_eval_steps = 100
    n_steps_before_eval = int(n_steps/10.) # eval every 10% of training
    n_steps_before_checkpoint = int(n_steps/5.) # checkpoint every 20% of
    train_percent = 0.9
    # learning_rate = 1e-2
    # learning_rate = 1e-3
    learning_rate = 3e-4

    # Model params
    assert (hparams.n_embed % hparams.n_heads == 0), (
            "n_embed must be divisible by n_heads")
    # Misc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_banner(f'Remote Device: {device} // GPU: {gpu}')

    #########################
    ### Data & Model prep ###
    #########################
    # Construct dataset
    dataset = Dataset('shakespeare_char.txt', train_percent,
                      batch_size, hparams.context_size, n_eval_steps, device)

    # Build Model
    model = AttentionModel(dataset.vocab_size, hparams, device)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_parameters = sum(p.numel() for p in model.parameters())
    print_banner(f"Num parameters: {num_parameters}")

    # Helper function for kicking off model generation
    def generate(model, n_new_tokens):
        gen_start = (
            dataset.newline_encoded * torch.ones((1, 1), dtype=torch.long))
        gen_start = gen_start.to(device)
        gen_out = model.generate(gen_start, n_new_tokens)[0] # 0th batch
        chars_out = dataset.decode([x for x in gen_out.tolist()])
        str_out = "".join(chars_out)
        return str_out

    # Make sure generate works before training.
    n_new_tokens = 100
    generate(model, n_new_tokens) # ignore return val

    ####################################
    ### Logging & Checkpointing prep ###
    ####################################
    # experiment_name is used twice intentionally below:
    # Put experiment in model name so we can see it on Tensorboard
    model_name = (f"{experiment_name}"
        f"_context={hparams.context_size}_n_heads={hparams.n_heads}"
        f"_dropout={hparams.dropout}")
    print_banner(model_name)
    # Create a experiment name folder so Tensorboard can group by experiment
    model_log_dir = log_path / f"{experiment_name}/{model_name}"
    os.makedirs(model_log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=f"{model_log_dir}/train")
    val_writer   = SummaryWriter(log_dir=f"{model_log_dir}/val")
    # Log hparams
    pretty_hparams_str = ""
    for k,v in hparams.__dict__.items():
        pretty_hparams_str += f"{k}: {v}\n"
    pretty_hparams_str += f"Num parameters: {num_parameters}"
    train_writer.add_text("Hyperparameters", pretty_hparams_str)

    # Load & Checkpointing code
    model_save_dir = save_path / model_name
    if model_save_dir.exists():
        print ("Loading model from checkpiont...")
        checkpoint = torch.load(str(model_save_dir / model_filename))
        model.load_state_dict(checkpoint['model'])
        start_step = checkpoint['steps'] + 1
    else:
        os.makedirs(model_save_dir, exist_ok=True)
        start_step = 0
        checkpoint = {
            'model': model.state_dict(),
            'chars': dataset.chars,
            'optimizer': optimizer.state_dict(),
            'val_loss': float('inf'),
            'steps': start_step,
            'hparams': hparams,
        }

    ################
    ### Training ###
    ################
    t_last = timer()
    for step in range(start_step, n_steps+1):
        # sample
        xb, yb = dataset.get_batch('train')

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # logging
        train_writer.add_scalar(f"Cross Entropy Loss", loss.item(), step)

        # eval
        if step % n_steps_before_eval == 0:
            out = dataset.eval_model(model)
            runtime_s = timer() - t_last
            print (f"{step:5d}) // {runtime_s:>5.2f}s"
                f" // Train Loss: {out['train']:.2f} // Val Loss:"
                f" {out['val']:.2f}")
            val_writer.add_scalar(f"Cross Entropy Loss", out['val'], step)
            t_last = timer()
            train_writer.flush()

        # checkpointing
        if step > 0 and step % n_steps_before_checkpoint == 0:
            print (f"Saving model to {model_save_dir}")
            checkpoint['steps'] = step
            checkpoint['val_loss'] = out['val']
            torch.save(checkpoint, model_save_dir / model_filename)
            if run_to_first_save:
                print ("Stopping early...")
                break

    # ##################
    # ### Generation ###
    # ##################
    # n_new_tokens = 1000
    # print ('')
    # print_banner("Sample Generation")
    # print (generate(model, n_new_tokens))
    return out['val'], hparams

######################################
### Model Inferece for Web Serving ###
######################################
@app.cls(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu)
class ModelInference:
    @modal.enter()
    def load_model(self):
        # Latest model:
        # checkpoint = torch.load(volume_path / model_filename)
        # Fixed to medium model out:
        checkpoint = torch.load(medium_model_path)
        hparams = checkpoint['hparams']

        chars = checkpoint['chars']
        vocab_size = len(chars)
        stoi = {c:i for i,c in enumerate(chars)}
        itos = {i:c for i,c in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: [itos[i] for i in l]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AttentionModel(vocab_size, hparams, self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)

    @modal.method()
    def generate(self, prompt):
        n_new_tokens = 1000
        encoded_prompt = self.encode(prompt)
        torch_input = torch.tensor(encoded_prompt, dtype=torch.long)
        torch_input = torch_input.view(1, len(torch_input)) # Batch dim.
        torch_input = torch_input.to(self.device)

        gen_out = self.model.generate(torch_input, n_new_tokens)[0] # 0th batch
        chars_out = self.decode([x for x in gen_out.tolist()])
        str_out = "".join(chars_out)
        return str_out

#######################
### Web Application ###
#######################
@app.function(image=image) # Why do I need this image? Torch again.
@modal.web_endpoint(method="POST")
def web_generate(item: dict):
    output = ModelInference().generate.remote(item['prompt'])
    return {'web_generate': output}
    # return {'web_generate': f"Hello {x}!"}

@app.function(
    image=image,
    concurrency_limit=3,
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text=""):
        if not text:
            text = example_prompts[0]
        # return text[::-1]
        return ModelInference().generate.remote(text)

    example_prompts = [
        f"Where art thou Lucas?",
        f"Were that it was",
        f"Macbeth, fair is foul, foul is fair, but who are you?",
        f"Brevity is the soul of wit, so what is the soul of foolishness?",
    ]


    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    # add a gradio UI around inference
    with gr.Blocks(
        theme=theme, css=css, title="Tiny LLM"
    ) as interface:
        # Title
        gr.Markdown(
            f"# Generate Shakespeare text using the prompt",
        )

        # Input and Output
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input:")
                inp = gr.Textbox(  # input text component
                    label="",
                    placeholder=f"Write some Shakespeare like text or keep it empty!",
                    lines=10,
                )
            with gr.Column():
                gr.Markdown("## Output:")
                out = gr.Textbox(  # output text component
                    label="",
                    lines=10,
                )

        # Button to trigger inference and a link to Modal
        with gr.Row():
            btn = gr.Button("Generate", variant="primary", scale=2)
            btn.click(
                fn=go, inputs=inp, outputs=out
            )  # connect inputs and outputs with inference function

            gr.Button(  # shameless plug
                " Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        # Example prompts
        with gr.Column(variant="compact"):
            # add in a few examples to inspire users
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

@app.local_entrypoint()
def main():
    experiment_name = f"E{datetime.now().strftime('%Y-%m%d-%H%M%S.%f')}"
    default_hparams = ModelHyperparameters()

    # Build list of hyperparameters to train & validate
    hparams_list = []
    # h_options = (1, default_hparams.n_heads)
    # c_options = (8, default_hparams.context_size)
    # d_options = (0.0, default_hparams.dropout)
    h_options = (default_hparams.n_heads,)
    c_options = (8, default_hparams.context_size,)
    d_options = (default_hparams.dropout,)
    for n_heads in h_options:
        for context_size in c_options:
            for dropout in d_options:
                hparams_list.append(ModelHyperparameters(
                    n_heads=n_heads,
                    context_size=context_size,
                    dropout=dropout))

    # Run training for each hyperparameter setting
    results = []
    stop_early = True
    for result in train_model.starmap(
        [(h, experiment_name, stop_early) for h in hparams_list]):
        results.append(result)
        print (f"\tSingle result: {result}")

    best_result = min(results, key=lambda x: x[0])
    print (f"Best Result: {best_result}")
    best_hparams = best_result[1]

    # Finish training with best hyperparameters
    train_model.remote(best_hparams, experiment_name)
