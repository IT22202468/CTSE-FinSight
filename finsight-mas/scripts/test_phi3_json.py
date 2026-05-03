# scripts/test_llm_json.py
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

prompt = """Classify this financial news as BULLISH, BEARISH, or NEUTRAL.
Article: "Apple reports record quarterly revenue, beating all analyst expectations."
Respond ONLY with valid JSON in this exact format:
{"label": "BULLISH", "confidence": 0.95, "reason": "one sentence"}
No other text."""

response = llm.invoke(prompt)
print("Raw response:", response.content)

try:
    # LLMs sometimes wrap JSON in markdown — strip it
    clean = response.content.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(clean)
    print("Parsed successfully:", data)
except json.JSONDecodeError as e:
    print("FAILED to parse JSON:", e)
    print("You will need fallback parsing in your tools.")
