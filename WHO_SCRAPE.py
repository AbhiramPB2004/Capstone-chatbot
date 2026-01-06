import requests
from bs4 import BeautifulSoup
import json

BASE = "https://www.who.int"
INDEX = "https://www.who.int/news-room/fact-sheets"

resp = requests.get(INDEX)
soup = BeautifulSoup(resp.text, "html.parser")

items = soup.select("#alphabetical-nav-filter p a")

links = []

for a in items:
    title = a.text.strip()
    url = BASE + a["href"]

    links.append({
        "title": title,
        "url": url
    })

print("Total fact sheets found:", len(links))

with open("who_fact_links.json", "w") as f:
    json.dump(links, f, indent=2)
