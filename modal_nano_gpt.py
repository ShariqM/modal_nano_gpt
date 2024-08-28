# ---
# Hyperparameter optimization of LLM Training with Modal
#
# Trains an GPT LLM model from scratch on Shakespeare text data.
#
# A single experiment trains multiple models with different hyperparameters.
# The experiment name is 'E' followed by a datetimestamp.
#
# Logging:
# volume/logs/E2024-01-01-000000.000000/
#   E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/train
#
# Model Checkpoints:
# volume/models/E2024-01-01-000000.000000/
#  E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/nano_gpt_model.pt
#
# There will best a symlink that links to the best hyperparameter model:
# volume/models/E2024-01-01-000000.000000/best_nano_gpt_model.pt
#
# When serving the model this symlink will be used.
# ---

# # LLM Training with Hyperparameter Optimization
#
# When you want an LLM tailored to your specific data there are three options
# The easiest is [Prompt Engineering](https://en.wikipedia.org/wiki/Prompt_engineering)
# but the quality of the results isn't very high. The next level is
# [fine-tuning](https://modal.com/docs/examples/llm-finetuning) an existing
# LLM on your dataset which more involved but allows for better
# results. The final level is training an LLM from scratch which is the most
# involved but allows for the highest caliber results. In addition you may
# be able to shrink the model considerably and save money on inference costs
# after training.
#
# In this example we will explore training from scratch. In fact, we will train
# 8 LLM's in parallel with different hyperparameters and then select the best
# one. Sound challenging? Modal makes it easy.
#
# ## Basic Setup
# First we will import modal, the fastapi for serving tenorboard, the torch
# LLM model, and a dataset class. The torch model is a nano GPT style model
# via Karpathy and available [here](https://github.com/ShariqM/modal_nano_gpt/blob/master/model.py).
# The dataset class will manage the Shakespeare text data and is available
# [here](https://github.com/ShariqM/modal_nano_gpt/blob/master/model.py).

import modal
from modal import Image, build, enter, method
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from model import AttentionModel, Dataset

# We'll use an A10G GPU for training which will keep costs under ~$1. Since
# The default timeout is only 5 minutes so let's increase it to 20 minutes.
gpu = "A10G"
timeout = 20 * 60  # 20 minutes


# ## Create a Volume
# Since we'll be coordinating training across multiple machines we'll use a
# single [distributed volume](https://modal.com/docs/guide/volumes)
# to store the dataset, checkpointed models, and logs.
volume = modal.Volume.from_name("nano_gpt_volume")
volume_path = Path("/vol/data")
model_filename = "nano_gpt_model.pt"
best_model_filename = "best_nano_gpt_model.pt"
log_path = volume_path / "logs"
save_path = volume_path / "models"

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"

# ## Define a container image
# The container image will be based on the latest Debian slim image with torch
# for training, gradio for serving a web interface, and tensorboard for
# monitoring training.
torch_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "gradio~=3.50.2",
        "tensorboard",
    )
)

app = modal.App("modal_nano_gpt")

with torch_image.imports():
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

# ## Training Function

# Here we will define the training function making sure to include the image,
# volume, gpu, and timeout parameters.

# Training consists of specificying optimization parameters, loading the
# dataset, building the model, setting up logging and checkpointing, and finally
# the training itself.
@app.function(
    image=torch_image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=timeout)
def train_model(hparams, experiment_name, run_to_first_save=False):
    ##########################################
    ### Optimization, Data, and Model prep ###
    ##########################################
    # Optimization, training, and validation parameters
    batch_size = 64
    n_steps = 5000
    n_eval_steps = 100
    n_steps_before_eval = int(n_steps/10.)  # eval every 10% of training
    n_steps_before_checkpoint = int(n_steps/5.)  # save every 20% of training
    train_percent = 0.9
    learning_rate = 3e-4

    # Use GPU if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_banner(f'Remote Device: {device} // GPU: {gpu}')

    input_file_path = volume_path /  'shakespeare_char.txt'
    volume.reload()  # Make sure we have the latest data.
    if not os.path.exists(input_file_path):
        print ("Downloading Shakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
        volume.commit()  # Commit to disk

    # Construct dataset
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = Dataset(text, train_percent, batch_size, hparams.context_size,
                      n_eval_steps, device)

    # Build Model
    model = AttentionModel(dataset.vocab_size, hparams, device)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_parameters = sum(p.numel() for p in model.parameters())
    print_banner(f"Num parameters: {num_parameters}")

    ####################################
    ### Logging & Checkpointing prep ###
    ####################################
    model_name = (f"{experiment_name}"
        f"_context_size={hparams.context_size}_n_heads={hparams.n_heads}"
        f"_dropout={hparams.dropout}")
    print_banner(model_name)

    # Save logs to something like:
    # volume/logs/E2024-01-01-000000.000000/
    #   E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/train
    model_log_dir = log_path / f"{experiment_name}/{model_name}"
    os.makedirs(model_log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=f"{model_log_dir}/train")
    val_writer   = SummaryWriter(log_dir=f"{model_log_dir}/val")

    # Save hyperparameters to tensorboard for easy reference
    pretty_hparams_str = ""
    for k,v in hparams.__dict__.items():
        pretty_hparams_str += f"{k}: {v}\n"
    pretty_hparams_str += f"Num parameters: {num_parameters}"
    train_writer.add_text("Hyperparameters", pretty_hparams_str)

    # Load & Save models to something like:
    # volume/models/E2024-01-01-000000.000000/
    #  E2024-01-01-000000.000000_context=8_n_heads=1_dropout=0.0/nano_gpt_model.pt
    model_save_dir = save_path / experiment_name / model_name
    if model_save_dir.exists():
        print ("Loading model from checkpiont...")
        checkpoint = torch.load(str(model_save_dir / model_filename))
        if run_to_first_save:
            # Already done. Someone restarted the job.
            print ("Stopping early...")
            return checkount['val_loss'], hparams
        else:
            # Create symlink to my (best) model so it's easy to find for web serving.
            os.symlink(str(model_save_dir / model_filename),
                        str(save_path / experiment_name / best_model_filename))
            volume.commit()  # Commit the symlink.

        model.load_state_dict(checkpoint['model'])
        start_step = checkpoint['steps'] + 1
    else:
        assert run_to_first_save, "should have loaded ckpt" # can remove later.
        os.makedirs(model_save_dir, exist_ok=True)
        start_step = 0
        # Save metadata for training restarts and inference
        checkpoint = {
            'model': model.state_dict(),
            'chars': dataset.chars,
            'optimizer': optimizer.state_dict(),
            'val_loss': float('inf'),
            'steps': start_step,
            'hparams': hparams,
            'finished_training': False,
        }

    ################
    ### Training ###
    ################
    t_last = timer()
    for step in range(start_step, n_steps+1):
        # sample a batch of data
        xb, yb = dataset.get_batch('train')

        # evaluate the loss, calculate & apply gradients
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log traiing loss
        train_writer.add_scalar(f"Cross Entropy Loss", loss.item(), step)

        # evaluate model on validation set
        if step % n_steps_before_eval == 0:
            out = dataset.eval_model(model)
            runtime_s = timer() - t_last
            print (f"{step:5d}) // {runtime_s:>5.2f}s"
                f" // Train Loss: {out['train']:.2f} // Val Loss:"
                f" {out['val']:.2f}")
            val_writer.add_scalar(f"Cross Entropy Loss", out['val'], step)
            t_last = timer()
            train_writer.flush()
            volume.commit()

        # save model with checkpoint information
        if step > 0 and step % n_steps_before_checkpoint == 0:
            print (f"Saving model to {model_save_dir}")
            checkpoint['finished_training'] = (
                step >= n_steps)  # Mark as finished if we hit n steps.
            checkpoint['steps'] = step
            checkpoint['val_loss'] = out['val']
            torch.save(checkpoint, model_save_dir / model_filename)
            volume.commit()
            if run_to_first_save:
                print ("Stopping early...")
                break

    return out['val'], hparams

# ## Main Entry Point
# The main entry point will run the hyperparameter optimization training.
# First, we'll specify the default hyperparameters for the model, taken from
# [Karpathy's biggest model](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5976s),
# which add up to 10 million total parameters.
@dataclass
class ModelHyperparameters:
    n_heads: int = 6
    n_embed: int = 384
    n_blocks: int = 6
    context_size: int = 256
    dropout: float = 0.2

# Next we will define the main entry point which will run the hyperparameter
# optimization. We will train 8 models with different hyperparameters, varying
# the number of heads, the context size (called the block size in Karpathy
# lingo), and the dropout rate.
#
# We will run the training for each model until the first checkpoint, and then
# check which one has the best validation loss. We will then finish training the
# best model and save it to the models directory.
@app.local_entrypoint()
def main():
    from datetime import datetime
    experiment_name = f"E{datetime.now().strftime('%Y-%m%d-%H%M%S.%f')}"
    default_hparams = ModelHyperparameters()

    # Build list of hyperparameters to train & validate
    hparams_list = []
    h_options = (1, default_hparams.n_heads)
    c_options = (8, default_hparams.context_size)
    d_options = (0.1, default_hparams.dropout)
    for n_heads in h_options:
        for context_size in c_options:
            for dropout in d_options:
                hparams_list.append(ModelHyperparameters(
                    n_heads=n_heads,
                    context_size=context_size,
                    dropout=dropout))

    # Run training for each hyperparameter setting
    results = []
    stop_early = True # stop early so we can compare val losses
    print (f"Testing {len(hparams_list)} hyperparameter settings")
    for result in train_model.starmap(
        [(h, experiment_name, stop_early) for h in hparams_list]):
        results.append(result)
        print (f"\tEarly stop val loss result: {result}")

    # Find the model and hparams with the lowest validation loss
    best_result = min(results, key=lambda x: x[0])
    print (f"Best early stop val loss result: {best_result}")
    best_hparams = best_result[1]

    # Finish training with best hparams
    train_model.remote(best_hparams, experiment_name)

# ## Bonus 1: Tensorboard Web App
# Similar to [this example](https://modal.com/docs/examples/flan_t5_finetune#:~:text=Monitoring%20Finetuning%20with-,Tensorboard,-Tensorboard%20is%20an)
# we will create a WSGI web app for serving the tensorboard logs of our model
# training.
@app.function(
    image=torch_image,
    volumes={volume_path: volume})
@modal.wsgi_app()
def monitor_training():
    import time
    print ("Tensorboard: Waiting 10 seconds for training to start...")
    time.sleep(10) # Wait for experiment folder to be created by training.
    volume.reload()  # Make sure we have the latest data.

    # Obtain the latest log path
    log_paths = glob.glob(f"{log_path}/*")
    latest_log_path = max(log_paths, key=os.path.getctime)
    monitor_path = Path(latest_log_path)
    print_banner(f"Monitoring: {monitor_path.name}")

    # Start tensorboard with the latest log path
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

# ## Bonus 2: Inference class for serving the model
# Similar to [this example](https://modal.com/docs/examples/dreambooth_app#:~:text=Running%20our-,model,-To%20generate%20images)
# we will create a class for running inference on the trained model.
#
# The idea will be to find the latest experiment that has best
# model checkpoint and to load that model for inference. In case
# training is ongiong, we will check for updated models on the fly
# and load them if available.
@app.cls(
    image=torch_image,
    volumes={volume_path: volume},
    gpu=gpu)
class ModelInference:
    def build_encode_decode(self, chars):
        # Create funcs for converting  text into digits (encode) and
        # vice versa (decode)
        stoi = {c:i for i,c in enumerate(chars)}
        itos = {i:c for i,c in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: [itos[i] for i in l]
        return encode, decode

    def load_model_impl(self):
        # Loop through all model dirs and load the latest available model
        save_model_dirs = glob.glob(f"{save_path}/*")
        sorted_model_dirs = sorted(
            save_model_dirs, key=os.path.getctime, reverse=True)
        found_model = False
        for latest_model_dir in sorted_model_dirs:
            if self.use_model_dir == latest_model_dir and self.is_fully_trained:
                return  # Already loaded
            print (f"Attemping to load from: {latest_model_dir} ...")
            try:
                checkpoint = torch.load(
                    f"{latest_model_dir}/{best_model_filename}")
                print ("Successfully loaded model.")
                found_model = True
                break
            except Exception as e:
                print (f"Error loading model: {e}")
        if not found_model:
            raise Exception(f"No models ready for serving.")

        # Model loaded successfully. Print info about the model
        self.use_model_dir = latest_model_dir
        hparams = checkpoint['hparams']
        chars = checkpoint['chars']
        steps = checkpoint['steps']
        val_loss = checkpoint['val_loss']
        self.is_fully_trained = checkpoint['finished_training']

        print (f"Loaded model with {steps} train steps "
            f" and val loss of {val_loss:.2f}"
            f" (fully_trained={self.is_fully_trained}")

        # Reconstruct encode/decode
        vocab_size = len(chars)
        self.encode, self.decode = self.build_encode_decode(chars)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AttentionModel(vocab_size, hparams, self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)

    @modal.enter()
    def load_model(self):
        self.use_model_dir = None
        self.is_fully_trained = False
        self.load_model_impl()

    @modal.method()
    def generate(self, prompt):
        self.load_model_impl() # Will load updated model if aviailable, o/w no op.

        # Generate 1000 new characters from input prompt
        n_new_tokens = 1000
        encoded_prompt = self.encode(prompt)
        # Create a torch tensor from the encoded prompt
        torch_input = torch.tensor(encoded_prompt, dtype=torch.long)
        torch_input = torch_input.view(1, len(torch_input)) # Add batch dim.
        torch_input = torch_input.to(self.device)

        # Generate new tokens
        gen_out = self.model.generate(torch_input, n_new_tokens)[0] # 0th batch
        # Decode from digits to text
        chars_out = self.decode([x for x in gen_out.tolist()])
        # Join the characters into a string and return
        str_out = "".join(chars_out)
        return str_out

# First, we will create a simple POST web endpoint for generating text.
# e.g.
# curl -X POST -H 'Content-Type: application/json' --data-binary '{"prompt":
# "\n"}' https://shariqm--modal-nano-gpt-web-generate-dev.modal.run
# Similar to [this example](https://modal.com/docs/examples/basic_web).
@app.function(image=torch_image)
@modal.web_endpoint(method="POST")
def web_generate(item: dict):
    output = ModelInference().generate.remote(item['prompt'])
    return {'web_generate': output}

# Second, we will create a Gradio web app for generating text. Similar to how
# it was done in the [Dreambooth example](https://modal.com/docs/examples/dreambooth_app#:~:text=Copy-,Wrap,-the%20trained%20model).
@app.function(
    image=torch_image,
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
            text = "\n"
        return ModelInference().generate.remote(text)

    example_prompts = [
        f"DUKE OF YORK:\nWhere art thou Lucas?",
        f"ROMEO:\nWhat is a man?",
        f"CLARENCE:\nFair is foul and foul is fair, but who are you?",
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
