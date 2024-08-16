# Train an LLM and serve it via FastAPI using Modal
LLM Training is based on [Karpathy's GPT from scratch tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5976s).

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
