from rag_helper import answer_question

while True:
    q = input("Ask (or exit): ")

    if q.lower() == "exit":
        break

    lang = input("Lang en/hi/kn: ")

    reply = answer_question(q, lang)

    print("\n GEMINI REPLY:\n", reply, "\n")
