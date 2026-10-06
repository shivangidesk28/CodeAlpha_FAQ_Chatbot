"""
app.py - the web server. It shows the chat page and answers chat requests.
Run it with:  python app.py
"""

from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot
from llm import ask_llm

app = Flask(__name__)
bot = FAQChatbot()


@app.route("/")
def home():
    return render_template("index.html", suggestions=bot.sample_questions(4))


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please type a message."}), 400

    message = message[:500]  # keep messages a sensible length
    result = bot.get_response(message)

    # No FAQ matched -> optionally ask the AI (only if a Gemini key is set)
    if result.get("source") == "none":
        ai_answer = ask_llm(message, bot.faqs)
        if ai_answer:
            result = {"answer": ai_answer, "confidence": 0.0, "matched_question": None,
                      "suggestions": [], "source": "ai"}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
