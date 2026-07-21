# FAQ & Troubleshooting

Common questions about using Meridian.AI. See the [Quick Start](QUICK_START.md) to get running and the [README](../README.md) for the full picture.

## Using the models

**Do I need a GPU?**
No. The model is deliberately compact (~430K parameters) and runs on CPU. Predictions just download a checkpoint from Hugging Face; you never train anything yourself.

**Do I need to train the model first?**
No. `hf_hub_download` pulls the latest checkpoint published by the hourly CI pipeline. The first call caches it locally, so later runs start instantly.

**Where do the checkpoints live?**
[meridianal/ARA.AI](https://huggingface.co/meridianal/ARA.AI) on Hugging Face — `models/Meridian.AI_Stocks.pt` and `models/Meridian.AI_Forex.pt`. They refresh every hour (stocks at `:00`, forex at `:30`).

**Which Python versions are supported?**
3.9 through 3.12.

## Interpreting predictions

**How far ahead does it predict?**
One day. `days=5` returns a 5-step recursive forecast, but error compounds quickly past day one — treat multi-day output as illustrative, not reliable.

**Can I trust the up/down direction?**
Treat it as a weak tilt, not a signal. Out of sample, neither the stock nor the forex model beats the always-up drift baseline on next-day direction. What *is* reliable is the predicted **magnitude**, which is calibrated at the zero-prediction error floor. See [Performance and Honest Expectations](../README.md#performance-and-honest-expectations).

**Why was the old 63.5% forex accuracy removed?**
It was a data artifact. Daily forex (`*=X`) candles are internally inconsistent — each bar's high/low spans a later window than its stored close, leaking the next close. Since 1.2.0, forex trains and evaluates with a one-day embargo that blocks the leak, and the honest number sits near the drift baseline.

## Troubleshooting

**`Checkpoint version too old` / load error.**
The loader only accepts architecture version 7.0 or newer. Delete any cached v5/v6 `.pt` files and re-download the current checkpoint.

**`hf_hub_download` fails or rate-limits.**
Retry — the Hub occasionally returns 429s. If it persists, check your network and that `huggingface_hub` is installed (`pip install -r requirements.txt`).

**Predictions look flat or identical across symbols.**
Make sure you passed a valid `model_path` from `hf_hub_download` and not the default placeholder path. A truly degenerate checkpoint would have been blocked by the sanity gate before publishing, so a flat output almost always means a wrong or missing model file.

**`torch` won't install.**
Install the CPU wheel explicitly first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.
