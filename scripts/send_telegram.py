import os

import requests
import yfinance as yf
from openai import OpenAI

# 1. Environment Secrets & Sanitization
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
raw_nvidia_key = os.environ.get("NVIDIA_API_KEY", "")

# Remove white-spaces, newlines, or invisible control characters
nvidia_key = "".join(raw_nvidia_key.split())

if not nvidia_key:
    raise ValueError("NVIDIA_API_KEY environment variable is missing!")

# 2. Extract Top Tickers from predictions.csv
top_stocks = []
if os.path.exists("predictions.csv"):
    with open("predictions.csv") as f:
        content = f.read()
        import re

        found = re.findall(r"\b[A-Z]{2,5}\b", content)
        ignored = {"MODEL", "PATH", "STOCK", "RANK", "DATE", "PREDICT", "INFO", "SYMBOL", "SCORE"}
        top_stocks = [t for t in found if t not in ignored][:5]

if not top_stocks:
    top_stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]

stocks_str = ", ".join(top_stocks)

# 3. Pull News via yfinance
news_context = ""
for ticker in top_stocks:
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news[:2] if hasattr(stock, "news") and stock.news else []
        if news_items:
            news_context += f"\nNews for {ticker}:\n"
            for item in news_items:
                title = item.get(
                    "title", item.get("content", {}).get("title", "Headline unavailable")
                )
                news_context += f"- {title}\n"
    except Exception as e:
        news_context += f"- Could not fetch news for {ticker}: {e}\n"

# 4. Generate Analysis via NVIDIA API
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)

prompt = f"""Ara AI evaluated the market universe and ranked these as top daily performers: {stocks_str}.

Using the news headlines below, explain what fundamental catalysts or market momentum might be driving these rankings. Keep the summary concise.

News:
{news_context}"""

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-r1",
    messages=[
        {"role": "system", "content": "You are a sharp financial analyst."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    stream=False,
)

analysis = completion.choices[0].message.content

# Remove potential DeepSeek thinking blocks if present in output
if "</think>" in analysis:
    analysis = analysis.split("</think>")[-1].strip()

# 5. Send Telegram Notification
message = f"📈 **Daily Quant Predictions** 📈\n\n**Top Picks:** {stocks_str}\n\n**AI Analysis:**\n{analysis}"

requests.post(
    f"https://api.telegram.org/bot{telegram_token}/sendMessage",
    json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
)
