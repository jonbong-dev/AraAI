import os
import re
import requests
import yfinance as yf
from openai import OpenAI

# 1. Environment Secrets
telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID')
nvidia_key = os.environ.get('NVIDIA_API_KEY')

if not nvidia_key:
    raise ValueError("NVIDIA_API_KEY environment variable is missing!")

# 2. Extract Top Tickers from prediction output
top_stocks = []
if os.path.exists('predictions.csv'):
    with open('predictions.csv', 'r') as f:
        content = f.read()
        # Find 3-5 letter uppercase ticker symbols in the output
        found = re.findall(r'\b[A-Z]{2,5}\b', content)
        # Exclude common system/command words
        ignored = {'MODEL', 'PATH', 'STOCK', 'RANK', 'DATE', 'PREDICT', 'INFO'}
        top_stocks = [t for t in found if t not in ignored][:5]

# Default fallback if parsing fails
if not top_stocks:
    top_stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]

stocks_str = ", ".join(top_stocks)

# 3. Pull News via yfinance
news_context = ""
for ticker in top_stocks:
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news[:3]
        if news_items:
            news_context += f"\nNews for {ticker}:\n"
            for item in news_items:
                title = item.get('title', item.get('content', {}).get('title', 'Headline unavailable'))
                news_context += f"- {title}\n"
    except Exception as e:
        news_context += f"- Could not fetch news for {ticker}: {e}\n"

# 4. Generate Analysis via NVIDIA DeepSeek
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_key
)

prompt = f"""Ara AI evaluated the market universe and ranked these as top daily performers: {stocks_str}.

Using the live news headlines below, explain what fundamental catalysts or market momentum might be driving these rankings. Keep the summary concise.

News:
{news_context}"""

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-pro-0813",
    messages=[
        {"role": "system", "content": "You are a sharp financial analyst."},
        {"role": "user", "content": prompt}
    ],
    temperature=1,
    top_p=0.95,
    max_tokens=2048,
    seed=42,
    extra_body={"chat_template_kwargs": {"thinking": False}},
    stream=False
)

analysis = completion.choices[0].message.content

# 5. Send Telegram Notification
message = f"📈 **Daily Quant Predictions** 📈\n\n**Top Picks:** {stocks_str}\n\n**DeepSeek Analysis:**\n{analysis}"

requests.post(
    f"https://api.telegram.org/bot{telegram_token}/sendMessage",
    json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
)
