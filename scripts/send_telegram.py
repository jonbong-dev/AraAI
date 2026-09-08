import os
import re
import requests
import yfinance as yf
from openai import OpenAI

# 1. Environment Secrets & Cleaning
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
raw_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
raw_nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
nvidia_key = "".join(raw_nvidia_key.split())

# Parse comma-separated Chat IDs into a list of clean strings
chat_ids = [c.strip() for c in raw_chat_id.split(",") if c.strip()]

print(f"Telegram Bot Token present: {bool(telegram_token)}")
print(f"Telegram Chat IDs found: {len(chat_ids)} ({chat_ids})")

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

# 4. Generate Analysis via NVIDIA API using OpenAI SDK
analysis = ""
if nvidia_key:
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key
        )

        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro-0813",
            messages=[
                {"role": "system", "content": "You are a sharp financial analyst."},
                {
                    "role": "user",
                    "content": f"Ara AI evaluated the market universe and ranked these as top daily performers: {stocks_str}.\n\nUsing the news headlines below, explain fundamental catalysts or market momentum driving these rankings.\n\nNews:\n{news_context}",
                },
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=2048,
            seed=42,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )

        analysis = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"NVIDIA SDK Exception: {e}")
        analysis = f"Market momentum currently favors {stocks_str} based on quantitative cross-sectional ranking."
else:
    analysis = f"Quantitative rankings generated for top holdings: {stocks_str}."

if "</think>" in analysis:
    analysis = analysis.split("</think>")[-1].strip()

# 5. Send Telegram Notification to Each Chat ID
message = (
    f"📈 DAILY QUANT PREDICTIONS 📈\n\n"
    f"Top Picks: {stocks_str}\n\n"
    f"AI Market Analysis:\n{analysis}"
)

if telegram_token and chat_ids:
    tg_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    
    for cid in chat_ids:
        payload = {"chat_id": cid, "text": message}
        try:
            response = requests.post(tg_url, json=payload, timeout=15)
            res_data = response.json()
            if res_data.get("ok"):
                print(f"Successfully sent Telegram message to Chat ID: {cid}")
            else:
                print(f"Failed sending to Chat ID {cid}: {res_data.get('description')}")
        except Exception as e:
            print(f"Error sending to Chat ID {cid}: {e}")
else:
    print("Telegram token or Chat IDs missing. Message sending skipped.")
