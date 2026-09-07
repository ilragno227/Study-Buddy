# 🧠 Study Buddy

A RAG-powered study assistant built with Streamlit. Made to make studying more fun for Gen Z/Alpha students. Upload your course PDFs and get a chat tutor grounded in your own material, auto-generated flashcards, and a multiple-choice quiz — with spoken answers in the chat.

## Features

- **💬 Chat** — ask questions about your material and get answers grounded only in the retrieved context, with an optional spoken voice response (Kokoro TTS)
- **🗂️ Flashcards** — generate a deck from your PDFs and flip through them one at a time with Previous / Show Answer / Next
- **📝 Quiz** — take a multiple-choice quiz at one of three difficulties (`EZ`, `Tuff`, `Charlie Kirk`), scored at the end with a pass/fail result

## Tech stack

| Piece | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Text generation | [Groq](https://groq.com) (`openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) |
| PDF parsing | [pypdf](https://github.com/py-pdf/pypdf) |
| Text-to-speech | [Kokoro](https://github.com/hexgrad/kokoro) |

## Usage

1. Upload a PDF in the sidebar.
2. Click **Build Study Buddy** to extract, chunk, and embed the material.
3. Use the **Chat**, **Flashcards**, or **Quiz** tab.

## Notes

- All answers are grounded in the retrieved PDF context — if the material doesn't cover something, StudyBuddy says so rather than guessing.
- Quiz scoring: you need at least 50% correct (among questions you actually answered) to win.
- Chat responses avoid markdown and emoji by design, since they may be read aloud by Kokoro.

## Credits
- Mohamed Mahmoud
- Merna Mohamed
- Ranya Farrag
- Youssef Abady
