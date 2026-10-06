# CodeAlpha_FAQ_Chatbot

A simple FAQ chatbot built for the CodeAlpha Artificial Intelligence internship (Task 2).

## What it does
- Loads FAQs (question + answer pairs) from `faqs.json`
- Cleans text with NLP: lowercase, tokenize, remove stopwords, stem (NLTK)
- Matches the user's question to the closest FAQ using TF-IDF and cosine similarity (scikit-learn)
- Shows the answer in a chat UI built with Flask, HTML, CSS and JavaScript
- OPTIONAL: if no FAQ matches, asks Google's Gemini AI (only when you add an API key; see below)

## How to run
1. Install Python 3.9 or newer
2. Open a terminal in this folder
3. (Optional) create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install the libraries: `pip install -r requirements.txt`
5. Start the app: `python app.py`
6. Open http://127.0.0.1:5000 in your browser

## Project structure
```
CodeAlpha_FAQ_Chatbot/
├── app.py            # Flask web server
├── chatbot.py        # NLP + matching logic
├── llm.py            # optional Gemini AI fallback
├── faqs.json         # the FAQ data (edit this to change the topic)
├── .env.example      # template for your secret API key
├── requirements.txt  # libraries needed
└── templates/
    └── index.html    # chat user interface
```

## Customize
Edit `faqs.json` to use your own questions and answers. Restart the app after saving.

## Optional: turn on the AI fallback (Gemini)
1. Get a free key at https://aistudio.google.com/apikey
2. Copy `.env.example` to a new file named `.env` and paste your key: `GEMINI_API_KEY=your_key`
3. `pip install -r requirements.txt`, then `python app.py`

Never upload your `.env` file to GitHub (`.gitignore` already excludes it).
Without a key, the bot works exactly the same, using only the FAQs.
