# Train an LLM and serve it via FastAPI using Modal
LLM Training is based on [Karpathy's GPT from scratch tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5976s).

# Data
The data is Shakespeare text from Project Gutenberg. Downloaded from
[here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

# Use
## Train model
modal run modal_nano_gpt

## Serve model
modal serve modal_nano_gpt

## Post to web app for generation
### Option 1
curl -X POST -H 'Content-Type: application/json' --data-binary '{"prompt": "\n"}' https://shariqm--modal-nano-gpt-web-generate-dev.modal.run
### Option 2
Visit https://shariqm--modal-nano-gpt-fastapi-app-dev.modal.run

## TODO
- Implement better tokenizer, e.g byte pair encoding (BPE)
- Use bfloat16 for faster training
- New datasets e.g. Erik's blog or OpenWebText
- Train a bigger model
- incorporate multiple GPUs
- Support loading OpenAI's GPT-2 model
- Incorporate better generation techniques e.g. nucleus sampling
