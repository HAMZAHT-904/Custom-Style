# GPT-2 Fine-tuning Lab

Fine-tune GPT-2 (or DistilGPT-2) on a small custom text dataset, then compare generations
from the base model and the fine-tuned model.

## Structure

```
backend/core.py    preprocessing, tokenizer, training loop, checkpoints, generation
backend/app.py     FastAPI server (dataset upload, training control, status, generation)
frontend/index.html  upload, training settings, loss chart, prompt, before/after results
data/campus_announcements.txt   starter dataset (30 short announcements)
cli.py             starter script: train + print before/after in the terminal
tests/             pytest test cases
checkpoints/       saved runs (created automatically)
```

## Setup and run

```
pip install -r requirements.txt
uvicorn backend.app:app --port 8000
```
Open http://localhost:8000, then: load the sample dataset, press **Start training**,
watch the loss chart, then press **Generate**.

The first run with `distilgpt2` or `gpt2` downloads the pretrained weights from Hugging Face
(needs internet once). Use `distilgpt2` on a CPU. Use `tiny-offline` only to check the pipeline:
it is a random 2-layer model, so its text is not meaningful.

Command line version:
```
python cli.py --data data/campus_announcements.txt --base distilgpt2 --epochs 3 --prompt "Attention students:"
```

## Dataset format
Plain `.txt`. Separate examples with a blank line, or put one example per line.
Each example gets an end-of-text token, then everything is cut into fixed-size token blocks.
10% of the token stream (at the end) is kept as validation data.

## Settings
| Setting | Meaning | Good start |
|---|---|---|
| Epochs | passes over the data | 3 to 5 |
| Batch size | blocks per step | 4 |
| Learning rate | AdamW, linear warm-up then decay | 5e-5 |
| Block size | tokens per training block | 64 |
| Max steps | hard cap, 0 = none | 0 |

A checkpoint is saved at the end of every epoch to `checkpoints/<run>/model`
(plus `meta.json` with settings and losses).

## Tests
```
python -m pytest tests -v
```
They use the offline tiny model, so they run in seconds without downloads.

| Test area | What is checked |
|---|---|
| Training data preparation | cleaning, example splitting, tokenizer round trip, train/val blocks, too-small data rejected |
| Fine-tuning | loss drops, validation loss logged per epoch and improves, stop button saves a checkpoint |
| Checkpoints | reloaded model gives the same logits; `meta.json` written |
| Loss monitoring | train loss per step, validation loss per epoch, all values finite |
| Custom data influence | fine-tuned output uses more words and keywords from your data than the base |
| API | dataset upload, validation errors, training, status polling, generation, missing run |

## How "custom data influence" is measured
For each generation, content words (stop words removed) are compared with your dataset:
**coverage** = share of generated content words that appear in your data;
**keyword hits** = how many of your data's top keywords appear. Compare these between the two columns.

## Notes
- Full GPT-2 fine-tuning needs a GPU for large datasets. With a few thousand tokens, `distilgpt2`
  on a laptop CPU is feasible; keep epochs low and the dataset small.
- With very small data the model can overfit quickly. Watch the validation dots in the loss chart:
  if they go up while training loss keeps falling, stop earlier or lower the learning rate.
