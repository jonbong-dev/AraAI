import os
import re

import requests
import yfinance as yf

# 1. Environment Secrets & Cleaning
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
raw_nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
nvidia_key = "".join(raw_nvidia_key.split())

print(f"Telegram Bot Token present: {bool(telegram_token)}")
print(f"Telegram Chat ID present: {bool(chat_id)}")

# 2. Extract Top Tickers from predictions.csv
top_stocks = []
if os.path.exists("predictions.csv"):
    with open("predictions.csv") as f:
        content = f.read()
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

# 4. Generate Analysis via Direct REST Call
analysis = ""
if nvidia_key:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a sharp financial analyst."},
            {
                "role": "user",
                "content": f"Ara AI evaluated the market universe and ranked these as top daily performers: {stocks_str}.\n\nUsing the news headlines below, explain fundamental catalysts or market momentum driving these rankings.\n\nNews:\n{news_context}",
            },
        ],
        "temperature": 0.5,
        "max_tokens": 1024,
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            analysis = data["choices"][0]["message"]["content"].strip()
        else:
            print(f"NVIDIA API Status {res.status_code}: {res.text}")
            analysis = f"Market momentum currently favors {stocks_str} based on quantitative cross-sectional ranking."
    except Exception as e:
        print(f"NVIDIA API Exception: {e}")
        analysis = f"Market momentum currently favors {stocks_str} based on quantitative cross-sectional ranking."
else:
    analysis = f"Quantitative rankings generated for top holdings: {stocks_str}."

if "</think>" in analysis:
    analysis = analysis.split("</think>")[-1].strip()

# 5. Send Telegram Notification (Plain Text to Avoid Parse Failures)
message = (
    f"📈 DAILY QUANT PREDICTIONS 📈\n\n"
    f"Top Picks: {stocks_str}\n\n"
    f"AI Market Analysis:\n{analysis}"
)

if telegram_token and chat_id:
    tg_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(tg_url, json=payload, timeout=15)
    print(f"Telegram API Status Code: {response.status_code}")
    print(f"Telegram API Response: {response.text}")
else:
    print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables are empty!")
