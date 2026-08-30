import os
import re
import requests
import yfinance as yf

# 1. Environment Secrets
telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
nvidia_key = "".join(os.environ.get('NVIDIA_API_KEY', '').split())

# 2. Extract Top Tickers from predictions.csv
top_stocks = []
if os.path.exists('predictions.csv'):
    with open('predictions.csv', 'r') as f:
        content = f.read()
        found = re.findall(r'\b[A-Z]{2,5}\b', content)
        ignored = {'MODEL', 'PATH', 'STOCK', 'RANK', 'DATE', 'PREDICT', 'INFO', 'SYMBOL', 'SCORE'}
        top_stocks = [t for t in found if t not in ignored][:5]

if not top_stocks:
    top_stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]

stocks_str = ", ".join(top_stocks)

# 3. Pull News via yfinance
news_context = ""
for ticker in top_stocks:
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news[:2] if hasattr(stock, 'news') and stock.news else []
        if news_items:
            news_context += f"\nNews for {ticker}:\n"
            for item in news_items:
                title = item.get('title', item.get('content', {}).get('title', 'Headline unavailable'))
                news_context += f"- {title}\n"
    except Exception as e:
        news_context += f"- Could not fetch news for {ticker}: {e}\n"

# 4. Generate Analysis via Direct REST Call (Avoids OpenAI 404 SDK Mappings)
analysis = ""
if nvidia_key:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Standard active NVIDIA NIM model identifier
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a sharp financial analyst."},
            {"role": "user", "content": f"Ara AI evaluated the market universe and ranked these as top daily performers: {stocks_str}.\n\nUsing the news headlines below, explain fundamental catalysts or market momentum driving these rankings.\n\nNews:\n{news_context}"}
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            analysis = data['choices'][0]['message']['content'].strip()
        else:
            print(f"NVIDIA API Returned Status {res.status_code}: {res.text}")
            analysis = f"Market momentum currently favors {stocks_str} based on quantitative cross-sectional ranking."
    except Exception as e:
        print(f"NVIDIA API Error: {e}")
        analysis = f"Market momentum currently favors {stocks_str} based on quantitative cross-sectional ranking."
else:
    analysis = f"Quantitative rankings generated for top holdings: {stocks_str}."

# Clean up thinking tags if DeepSeek or R1 reasoning models are used
if "</think>" in analysis:
    analysis = analysis.split("</think>")[-1].strip()

# 5. Send Telegram Notification
message = f"📈 **Daily Quant Predictions** 📈\n\n**Top Picks:** {stocks_str}\n\n**AI Market Analysis:**\n{analysis}"

if telegram_token and chat_id:
    requests.post(
        f"https://api.telegram.org/bot{telegram_token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    )
    print("Telegram notification sent successfully.")
else:
    print("Telegram credentials missing, skipping dispatch.")
