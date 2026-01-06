from google import genai
import json
import numpy as np


API_KEY = "AIzaSyCffZofrFQ15YxYGMZPBEJzmEj7rrLFnWg"          # <-- change this
LLM_MODEL = "models/gemini-2.5-flash"
EMB_MODEL = "models/text-embedding-004"

client = genai.Client(api_key=API_KEY)


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

    # embed query
    res = client.models.embed_content(
        model=EMB_MODEL,
        contents=query
    )

    qvec = res.embeddings[0].values

    scored = []

    for item in VECTOR_DB:
        score = cosine(qvec, item["embedding"])
        scored.append((score, item["content"]))

    scored.sort(reverse=True)
    

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
Answer the health question using ONLY the context below.if its greeting respond with greeting
donot only refrain to the context if you arre getting any medically related questions try to answer them. but maximise 
the use of the contexxt. only answer question related to medical questions on disease any other questions pls respon with cannot answer that.


Context:
{context}

Question:
{english}

Answer clearly. Avoid speculation. If unsure, say so.
"""

    res = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    answer_en = "".join(p.text for p in res.candidates[0].content.parts)

    # 4) translate back if needed
    return translate_from_english(answer_en, lang)
