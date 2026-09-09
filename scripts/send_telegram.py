import os
import re

import requests
import yfinance as yf
from openai import OpenAI

# 1. Environment Secrets & Parsing
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
raw_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
raw_nvidia_key = os.environ.get("NVIDIA_API_KEY", "")

# Clean stray quotes/spaces
nvidia_key = raw_nvidia_key.strip().strip('"').strip("'")

# Parse comma-separated Chat IDs: ['485686834', '936673392']
chat_ids = [c.strip() for c in raw_chat_id.split(",") if c.strip()]

print(f"Telegram Bot Token present: {bool(telegram_token)}")
print(f"Telegram Chat IDs found: {len(chat_ids)}")

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

# 3. Pull News via yfinance (Capped to 1,500 characters max)
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

# Prevent news payload from bloating the LLM prompt
news_context = news_context[:1500]

# 4. Generate Analysis via NVIDIA API
analysis = ""
if nvidia_key:
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)

        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro-0813",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sharp financial analyst. Provide a brief, high-impact market summary under 1500 characters.",
                },
                {
                    "role": "user",
                    "content": f"Ara AI evaluated the market universe and ranked these as top daily performers: {stocks_str}.\n\nUsing the news headlines below, explain fundamental catalysts driving these rankings in 2 brief paragraphs.\n\nNews:\n{news_context}",
                },
            ],
            temperature=0.7,
            top_p=0.95,
            max_tokens=500,  # Strict limit on response size
            seed=42,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False,
        )

        analysis = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"NVIDIA SDK Exception: {e}")
        analysis = f"Market momentum currently favors {stocks_str} based on quantitative cross-sectional ranking."
else:
    analysis = f"Quantitative rankings generated for top holdings: {stocks_str}."

# Clean out any DeepSeek thinking/reasoning blocks
if "</think>" in analysis:
    analysis = analysis.split("</think>")[-1].strip()


# 5. Safe Multi-Part Sending Function
def send_safe_telegram_messages(bot_token, target_chat_id, header, body_text):
    tg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # If total length is under 3800 chars, send in one message
    full_text = f"{header}\n\n{body_text}"
    if len(full_text) <= 3800:
        messages_to_send = [full_text]
    else:
        # Send header/picks first, then chunk the analysis
        messages_to_send = [header]
        chunk_size = 3500
        for i in range(0, len(body_text), chunk_size):
            messages_to_send.append(body_text[i : i + chunk_size])

    for idx, msg in enumerate(messages_to_send):
        payload = {"chat_id": target_chat_id, "text": msg}
        try:
            response = requests.post(tg_url, json=payload, timeout=15)
            res_data = response.json()
            if res_data.get("ok"):
                print(
                    f"Successfully sent message part {idx+1}/{len(messages_to_send)} to Chat ID: {target_chat_id}"
                )
            else:
                print(
                    f"Failed sending part {idx+1} to {target_chat_id}: {res_data.get('description')}"
                )
        except Exception as e:
            print(f"Error sending part {idx+1} to {target_chat_id}: {e}")


# Trigger sending
header_text = f"📈 DAILY QUANT PREDICTIONS 📈\n\nTop Picks: {stocks_str}"
analysis_text = f"AI Market Analysis:\n{analysis}"

if telegram_token and chat_ids:
    for cid in chat_ids:
        send_safe_telegram_messages(telegram_token, cid, header_text, analysis_text)
else:
    print("Telegram token or Chat IDs missing. Message sending skipped.")
