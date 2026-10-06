"""
chatbot.py - the "brain" of the FAQ chatbot.

How it works (in simple words):
1. Load the FAQs from faqs.json. Each FAQ has a question, an answer and an
   optional list of "keywords" (other ways people might ask the same thing).
2. Every question and every keyword phrase becomes one "pattern" that points
   back to its FAQ.
3. Clean each pattern: lowercase -> split into words -> remove filler words
   (stopwords) -> reduce words to their root (stemming, using NLTK).
4. Turn the cleaned patterns into numbers using TF-IDF.
5. When a user types something, clean it the same way and compare it with
   every pattern using cosine similarity.
6. An FAQ is accepted only if
      (a) its best pattern is similar enough (SIMILARITY_THRESHOLD), and
      (b) that pattern contains most of the user's meaningful words
          (MIN_COVERAGE, 60%). This stops one shared word (like "store") from
          triggering a wrong answer.
   Otherwise the bot says it didn't understand and suggests related questions.
"""

import json
import re
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

# The best FAQ must score at least this much (0 to 1) to be used as the answer.
SIMILARITY_THRESHOLD = 0.45
# At least this share of the user's meaningful words must appear in the matched pattern.
# (With 2 meaningful words, both must match; with 3, at least 2; and so on.)
MIN_COVERAGE = 0.6

# A fixed list of English filler words. It is built in on purpose, so the bot
# behaves the same on every computer (no internet or NLTK download needed).
STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can could couldn d did didn do does doesn
doing don down during each few for from further had hadn has hasn have haven having
he her here hers herself him himself his how i if in into is isn it its itself just
ll m ma me might more most mustn my myself need needn no nor not now o of off on once
only or other our ours ourselves out over own re s same shan she should shouldn so
some such t than that the their theirs them themselves then there these they this
those through to too under until up ve very was wasn we were weren what when where
which while who whom why will with won would wouldn y you your yours yourself
yourselves please tell want know get let also may much many come accept give
""".split())

# Different words that mean the same thing are turned into one word, so
# "bill" and "invoice" (or "parcel" and "order") are treated alike.
SYNONYMS = {
    "bill": "invoice", "receipt": "invoice",
    "abroad": "international", "overseas": "international", "foreign": "international",
    "internationally": "international", "worldwide": "international",
    "broken": "damaged", "defective": "damaged", "faulty": "damaged",
    "parcel": "order", "package": "order", "shipment": "order",
}

GREETING_KEYS = {"hi", "hii", "hello", "hey", "hola", "namaste", "yo"}
GREETING_FILLER = {"good", "morning", "afternoon", "evening", "there", "bot", "again"}
THANKS_KEYS = {"thanks", "thank", "thx", "ty", "thankyou"}
THANKS_FILLER = {"you", "so", "much", "a", "lot", "very", "ok", "okay"}
BYE_KEYS = {"bye", "goodbye", "cya"}
BYE_FILLER = {"see", "you", "later", "take", "care"}


class FAQChatbot:
    def __init__(self, faq_path=None):
        faq_path = Path(faq_path) if faq_path else BASE_DIR / "faqs.json"
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        if not self.faqs:
            raise ValueError("faqs.json is empty - add at least one question and answer.")

        self.stemmer = PorterStemmer()

        # Step 1: build patterns (the question + each keyword phrase of every FAQ)
        self.pattern_faq = []     # which FAQ each pattern belongs to
        self.pattern_text = []    # the cleaned text of each pattern
        for faq_id, item in enumerate(self.faqs):
            for text in [item["question"]] + list(item.get("keywords", [])):
                cleaned = self.preprocess(text)
                if cleaned:
                    self.pattern_faq.append(faq_id)
                    self.pattern_text.append(cleaned)
        self.pattern_words = [set(t.split()) for t in self.pattern_text]

        # Step 2: learn the vocabulary and convert patterns into number vectors
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        self.pattern_matrix = self.vectorizer.fit_transform(self.pattern_text)

    # ------------------------------------------------------------------ #
    # Text cleaning (NLP preprocessing)
    # ------------------------------------------------------------------ #
    def preprocess(self, text):
        """lowercase -> tokenize -> remove stopwords -> unify synonyms -> stem."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        tokens = [t for t in tokens if t not in STOPWORDS]
        tokens = [SYNONYMS.get(t, t) for t in tokens]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    # ------------------------------------------------------------------ #
    # Small talk
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_small_talk(tokens, keys, filler):
        words = set(tokens)
        return bool(words) and bool(words & keys) and words <= (keys | filler)

    def small_talk(self, message):
        tokens = re.findall(r"[a-z0-9]+", message.lower())
        if self._is_small_talk(tokens, GREETING_KEYS, GREETING_FILLER):
            return "Hello! 👋 I'm ShopBot. Ask me anything about orders, delivery, returns or payments."
        if self._is_small_talk(tokens, THANKS_KEYS, THANKS_FILLER):
            return "You're welcome! 😊 Is there anything else I can help you with?"
        if self._is_small_talk(tokens, BYE_KEYS, BYE_FILLER):
            return "Goodbye! 👋 Have a great day and happy shopping!"
        return None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_faqs(self, cleaned):
        """Return two arrays with one number per FAQ:
        raw   - best similarity of any pattern (used for suggestions)
        valid - same, but only for patterns that also cover enough of the
                user's words (used to decide if we can answer)."""
        raw = np.zeros(len(self.faqs))
        valid = np.zeros(len(self.faqs))
        query_words = set(cleaned.split())
        if not query_words:
            return raw, valid

        sims = cosine_similarity(self.vectorizer.transform([cleaned]), self.pattern_matrix).flatten()
        for k, sim in enumerate(sims):
            if sim <= 0:
                continue
            faq_id = self.pattern_faq[k]
            raw[faq_id] = max(raw[faq_id], sim)
            coverage = len(query_words & self.pattern_words[k]) / len(query_words)
            if coverage >= MIN_COVERAGE:
                valid[faq_id] = max(valid[faq_id], sim)
        return raw, valid

    # ------------------------------------------------------------------ #
    # Main function: find the best answer
    # ------------------------------------------------------------------ #
    def get_response(self, message):
        message = (message or "").strip()
        if not message:
            return {"answer": "Please type a question and I'll do my best to help.",
                    "confidence": 0.0, "matched_question": None, "suggestions": [],
                    "source": "empty"}

        chit_chat = self.small_talk(message)
        if chit_chat:
            return {"answer": chit_chat, "confidence": 1.0,
                    "matched_question": None, "suggestions": [], "source": "smalltalk"}

        raw, valid = self.score_faqs(self.preprocess(message))
        best = int(valid.argmax())

        if valid[best] >= SIMILARITY_THRESHOLD:
            return {
                "answer": self.faqs[best]["answer"],
                "confidence": round(float(valid[best]), 3),
                "matched_question": self.faqs[best]["question"],
                "suggestions": [],
                "source": "faq",
            }

        # Not sure: suggest the closest questions (or a few popular ones)
        top_ids = [int(i) for i in raw.argsort()[::-1][:3] if raw[i] > 0.1]
        suggestions = [self.faqs[i]["question"] for i in top_ids] or self.sample_questions(3)
        return {
            "answer": "Sorry, I couldn't find an answer to that. 🤔 I can help with orders, "
                      "delivery, returns, payments and your account. Try rephrasing, "
                      "or pick one of these questions:",
            "confidence": round(float(raw.max()), 3),
            "matched_question": None,
            "suggestions": suggestions,
            "source": "none",   # no FAQ matched (app.py may ask the AI fallback)
        }

    def sample_questions(self, n=4):
        """Return the first n FAQ questions (used for the quick-reply buttons)."""
        return [item["question"] for item in self.faqs[:n]]


if __name__ == "__main__":
    # Quick test in the terminal: python chatbot.py
    bot = FAQChatbot()
    print("ShopBot is ready! Type 'quit' to exit.\n")
    while True:
        text = input("You: ")
        if text.strip().lower() in {"quit", "exit"}:
            break
        print("Bot:", bot.get_response(text)["answer"], "\n")
