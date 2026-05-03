# scripts/verify_setup.py
import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  v {label}: {result}")
    except Exception as e:
        print(f"  x {label}: FAILED — {e}")
        sys.exit(1)

print("FinSight MAS — Setup Verification\n")

if sys.version_info >= (3, 14):
    print("  x Python version: unsupported on Python 3.14+; use Python 3.13 or 3.12")
    sys.exit(1)

check("Python version", lambda: sys.version.split()[0])
check("crewai import", lambda: __import__("crewai") and "OK")
check("yfinance AAPL price", lambda: f"${__import__('yfinance').Ticker('AAPL').fast_info.last_price:.2f}")
check("feedparser", lambda: f"{len(__import__('feedparser').parse('https://finance.yahoo.com/rss/topstories').entries)} articles from Yahoo")
check("Ollama/phi3:mini", lambda: __import__('langchain_ollama').ChatOllama(model='phi3:mini').invoke('Reply OK').content.strip())
check("SQLite", lambda: __import__('sqlalchemy').create_engine('sqlite:///test.db') and "OK")

print("\nAll systems go. You are ready to build.\n")
