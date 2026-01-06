from google import genai
import json

API_KEY = "AIzaSyCffZofrFQ15YxYGMZPBEJzmEj7rrLFnWg"
MODEL = "models/text-embedding-004"

INPUT_FILE = "who_disease_articles.json"
OUTPUT_FILE = "health_vectors.json"

client = genai.Client(api_key=API_KEY)

with open(INPUT_FILE) as f:
    DATA = json.load(f)

records = []

for i, item in enumerate(DATA):

    title = item.get("title", "").strip()
    url = item.get("url", "").strip()
    content = item.get("content", "").strip()

    # Combine fields to embed
    text = f"{title}\n\n{content}".strip()

    if not text:
        print(f"[{i+1}] ⚠️ Skipping empty record")
        continue

    print(f"[{i+1}/{len(DATA)}] Embedding: {title[:80]}")

    res = client.models.embed_content(
        model=MODEL,
        contents=text
    )

    emb = res.embeddings[0].values  # <-- correct new SDK field

    records.append({
        "title": title,
        "url": url,
        "content": content,
        "embedding": emb
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(records, f, indent=2)

print("\nSaved →", OUTPUT_FILE)
