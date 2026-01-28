from google import genai
import json
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


API_KEY = "AIzaSyCOXbrcSKVDeeijk5L56-aYwPlzk5oWiug"          # <-- change this
LLM_MODEL = "models/gemini-2.5-flash"


client = genai.Client(api_key=API_KEY)

import re

def clean_text(text):
    # remove markdown symbols
    text = re.sub(r"[*#>`_~\-]", "", text)

    # remove extra blank lines
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()




# ---------- Load Vector DB ----------
with open("health_vectors.json") as f:
    VECTOR_DB = json.load(f)



# ---------- Cosine Similarity ----------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



# ---------- Translate to English ----------
def translate_to_english(text):

    prompt = f"""
Detect the language. Translate this into simple English only.
Text:
{text}
"""

    res = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return "".join(p.text for p in res.candidates[0].content.parts)



# ---------- Translate Back ----------
def translate_from_english(text, lang):

    if lang == "en":
        return text

    prompt = f"""
Translate the following text into {lang}.
Keep meaning same. Do NOT add information.

{text}
"""

    res = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return "".join(p.text for p in res.candidates[0].content.parts)



# ---------- Vector Search ----------
def vector_search(query, top_k=3):

    qvec = embedder.encode(query)

    scored = []

    for item in VECTOR_DB:
        score = cosine(qvec, item["embedding"])
        scored.append((score, item["content"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in scored[:top_k]]




# ---------- Main Answer Function ----------
def answer_question(user_text, lang="en"):

    # 1) normalize to english
    english = translate_to_english(user_text)

    # 2) retrieve similar answers
    context = "\n".join(vector_search(english))
    print("CONTEXT : " ,context)
    if not context.strip():
        context = "No related info found."

    # 3) generate response
    prompt = f"""
You are a friendly medical assistant chatting on Telegram.

Rules:
- Answer ONLY medical / disease questions.
- If greeting, greet politely.
- Use simple human language.
- NO markdown.
- NO bullet points.
- NO headings.
- Short paragraphs.
- Friendly tone.

Use context heavily.

Context:
{context}

User Question:
{english}

Reply like a caring doctor chatting on Telegram.
"""


    res = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    answer_en = "".join(p.text for p in res.candidates[0].content.parts)

    # Clean formatting
    answer_en = clean_text(answer_en)

    # Translate back
    final = translate_from_english(answer_en, lang)

    # Clean again after translation
    final = clean_text(final)

    return final

