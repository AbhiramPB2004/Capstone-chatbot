from google import genai
import os

API_KEY = 'AIzaSyBW9TOSuLhEwu859ABlrTnp4tvwvA3mOGc'

client = genai.Client(api_key=API_KEY)

MODEL = "models/gemini-2.5-flash"

def ask_gemini(question, lang="en"):

    prompt = f"Reply in {lang}. Answer clearly:\n{question}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt   
    )

    text = "".join(
        getattr(part, "text", "")
        for part in response.candidates[0].content.parts
    )

    return text


if __name__ == "__main__":
    while True:
        q = input("Ask (or exit): ")
        if q.lower() == "exit":
            break
        lang = input("Lang en/hi/kn: ")
        print("\n", ask_gemini(q, lang), "\n")
