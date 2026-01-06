import requests
from bs4 import BeautifulSoup
import json
import time

BASE = "https://www.who.int"

with open("who_fact_links.json") as f:
    links = json.load(f)

data = []

for i, item in enumerate(links):
    print(f"[{i+1}/{len(links)}] {item['title']}")

    try:
        r = requests.get(item["url"])
        soup = BeautifulSoup(r.text, "html.parser")

        # 🔥 THIS grabs the full article content
        article = soup.find("article", class_="sf-detail-body-wrapper")

        if not article:
            print(" -> ❌ No article found")
            continue

        # keep text clean but structured
        content = article.get_text(separator="\n", strip=True)

        data.append({
            "title": item["title"],
            "url": item["url"],
            "content": content
        })

    except Exception as e:
        print(" -> ERROR:", e)

    time.sleep(1)  # respect WHO servers

with open("who_disease_articles.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved records:", len(data))
