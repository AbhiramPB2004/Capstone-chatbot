from sentence_transformers import SentenceTransformer
import json

INPUT_FILE = "who_disease_articles.json"
OUTPUT_FILE = "health_vectors.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(INPUT_FILE) as f:
    DATA = json.load(f)

records = []

for i, item in enumerate(DATA):

    title = item.get("title", "").strip()
    url = item.get("url", "").strip()
    content = item.get("content", "").strip()

    text = f"{title}\n\n{content}".strip()

    if not text:
        print(f"[{i+1}] Skipping empty")
        continue

    print(f"[{i+1}/{len(DATA)}] Embedding: {title[:60]}")

    emb = model.encode(text).tolist()

    records.append({
        "title": title,
        "url": url,
        "content": content,
        "embedding": emb
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(records, f, indent=2)

print("\nSaved →", OUTPUT_FILE)
