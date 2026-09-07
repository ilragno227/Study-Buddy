"""
Study Buddy — Streamlit App
============================
Chat, Flashcards, and Quiz tabs, backed by Groq for generation (no local
model, no GPU needed — works fine on Streamlit Cloud's free tier) and
Kokoro for spoken answers in the Chat tab.

Needs two things set up before it'll run:
  1. A Groq API key: GROQ_API_KEY in Streamlit secrets or as an env var.
     Get one free at https://console.groq.com/keys
  2. A packages.txt file (repo root) containing the line "espeak-ng" —
     Kokoro TTS needs it as a system dependency, not just a pip package.
"""

import json
import os
import re
import tempfile

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────
# Config & constants
# ──────────────────────────────────────────────────────────────────────────

# gpt-oss-120b is a good default: strong quality, generous context, fast on
# Groq's hardware. Swap to "openai/gpt-oss-20b" for lower latency or
# "qwen/qwen3.6-27b" as another option — check console.groq.com/docs/models
# for what's currently available.
GENERATION_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TTS_VOICE = "af_heart"
TTS_SPEED = 1.05
TTS_SAMPLE_RATE = 24000

MAX_QUIZ_QUESTIONS = 20

# Direct Tenor media URLs (not the tenor.com/view/... page links) so they
# render inline via st.image().
WIN_IMAGE_URL = "https://media1.tenor.com/m/FpHhGgR4zvgAAAAC/social-credit-credit.gif"
LOSE_IMAGE_URL = "https://media1.tenor.com/m/F-D5EhlQXdMAAAAC/nalog.gif"

WIN_MESSAGE = "You Win, gg wp"
LOSE_MESSAGE = "You Lose, train harder twin!"

QUIZ_MODES = {
    "EZ": """
Generate easy questions.
Focus on:
- definitions
- direct facts
- basic concepts
- simple understanding

Questions should be straightforward and answerable directly from the provided material.
""",
    "Tuff": """
Generate medium-difficulty questions.
Focus on:
- understanding concepts
- comparisons
- relationships between ideas
- simple applications
- moderate reasoning

Avoid questions that are simply copied word-for-word from the material.
""",
    "Charlie Kirk": """
Generate very difficult questions.
Focus on:
- deep understanding
- subtle distinctions
- applying concepts
- multi-step reasoning
- distinguishing between closely related ideas
- plausible but incorrect distractors

Do NOT make questions difficult by using obscure information.
Make them difficult because they require genuine understanding of the material.
""",
}

FLASHCARD_PROMPT_TEMPLATE = """You are helping a student build study flashcards from their course material.

Using ONLY the material below, generate {num_cards} flashcards that cover the
most important concepts, definitions, and facts. Each flashcard must have a
short, clear "question" (or term) and a concise "answer".

MATERIAL:
{context}

Respond with ONLY a valid JSON array, no other text, no markdown code fences.
Format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""

# Chat answers may be spoken aloud by Kokoro, so the system prompt bans
# markdown and emoji outright — both read terribly through TTS and there's
# no clean way to strip only "the bad parts" after the fact.
SYSTEM_PROMPT = """
IDENTITY

You are StudyBuddy, a sharp and encouraging AI study coach. A student
has handed you material they're trying to learn, and your job is to make
it click for them - not recite it back.

Every answer you give may be read out loud by a text-to-speech voice, in
addition to being shown as text. Because of that:
- Never use markdown symbols - no **, ##, -, *, `, > or bullet dashes.
- Never use emoji.
Write the way a real tutor talks out loud: plain sentences, natural
spoken structure, nothing that only makes sense on a screen.

GROUNDING RULES

1. Answer using only the retrieved context you're given. Do not add
   outside facts, even ones you're confident are true.
2. If the context doesn't contain the answer, say so plainly.
3. You may reference earlier turns in the conversation for continuity,
   but the retrieved context is always the source of truth for facts.

HOW YOU TEACH

- Lead with plain language, then introduce the technical term once the
  idea already makes sense.
- Walk through processes as a spoken sequence - first this, then that -
  never as a numbered or bulleted list.
- Keep answers tight: a few short spoken paragraphs by default.
- Never say "as an AI" or reference being a language model.
"""

# ──────────────────────────────────────────────────────────────────────────
# Page config & styling (matches the Study Buddy logo: olive green + cream)
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Study Buddy", page_icon="🧠", layout="wide")

# ──────────────────────────────────────────────────────────────────────────
# API key resolution
# ──────────────────────────────────────────────────────────────────────────


def resolve_api_key():
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
        if secret_key:
            return secret_key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# ──────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ──────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_resource(show_spinner="Connecting to Groq...")
def load_llm(api_key):
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="Loading Kokoro TTS...")
def load_tts():
    from kokoro import KPipeline
    return KPipeline(lang_code="a")


def generate_with_llm(prompt, max_new_tokens=1024, temperature=0.7, reasoning_effort="low"):
    client = st.session_state.get("llm")
    if client is None:
        st.error("Groq isn't connected yet — enter an API key and click Build Study Buddy.")
        st.stop()

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        reasoning_effort=reasoning_effort,
    )
    return response.choices[0].message.content.strip()


def clean_for_speech(text):
    text = re.sub(r"[*_#>`]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_speech(text, filename):
    import soundfile as sf

    tts = st.session_state.get("tts")
    if tts is None:
        return None

    text = clean_for_speech(text)
    if not text:
        return None

    audio_parts = []
    for _, _, audio in tts(
        text,
        voice=TTS_VOICE,
        speed=TTS_SPEED,
        split_pattern=r"(?<=[.!?])\s+",
    ):
        audio_parts.append(np.asarray(audio, dtype=np.float32))

    if not audio_parts:
        return None

    audio = np.concatenate(audio_parts)
    sf.write(filename, audio, TTS_SAMPLE_RATE)
    return filename


# ──────────────────────────────────────────────────────────────────────────
# PDF processing pipeline
# ──────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(path, source_name):
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"source": source_name, "page": page_number + 1, "text": text})
    return pages


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "page": page["page"], "source": page["source"]})
            start += chunk_size - overlap
    return chunks


def build_index(uploaded_files):
    embedding_model = load_embedding_model()

    all_pages = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        pages = extract_text_from_pdf(tmp_path, f.name)
        os.remove(tmp_path)
        for p in pages:
            p["text"] = clean_text(p["text"])
        all_pages.extend(pages)

    chunks = create_chunks(all_pages)
    if not chunks:
        return None, None, "Couldn't extract any text from the uploaded PDF(s)."

    # Batch-encode all chunks in one call rather than one at a time —
    # much faster than looping per-chunk for anything but tiny documents.
    embeddings = embedding_model.encode(
        [c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    status = f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s). Ready!"
    return chunks, index, status


def embed_query(query):
    embedding_model = load_embedding_model()
    embedding = embedding_model.encode(query, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embedding.reshape(1, -1))
    return embedding


def retrieve_relevant_chunks(query, chunks, index, top_k=5):
    query_embedding = embed_query(query)
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "source": chunks[idx]["source"],
                "score": float(score),
            }
        )
    return results


def build_context(results):
    parts = [f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}" for r in results]
    return "\n\n".join(parts)


def study_buddy_answer(question, conversation_history, chunks, index, top_k=5):
    results = retrieve_relevant_chunks(question, chunks, index, top_k=top_k)
    context = build_context(results)

    history = ""
    for message in conversation_history:
        history += f"\nStudent: {message['question']}\nStudyBuddy: {message['answer']}\n"

    prompt = f"""
{SYSTEM_PROMPT}

PREVIOUS CONVERSATION:
{history}

RETRIEVED CONTEXT:
{context}

CURRENT STUDENT QUESTION:
{question}

Answer the student naturally while staying grounded in the retrieved context.
"""
    return generate_with_llm(prompt)


# ──────────────────────────────────────────────────────────────────────────
# Flashcards
# ──────────────────────────────────────────────────────────────────────────


def generate_flashcards(chunks, num_cards=8, sample_chunks=12):
    step = max(1, len(chunks) // sample_chunks)
    sample = chunks[::step][:sample_chunks]
    context = "\n\n".join(c["text"] for c in sample)

    prompt = FLASHCARD_PROMPT_TEMPLATE.format(num_cards=num_cards, context=context)
    max_tokens = min(8000, 150 * num_cards + 400)
    raw = generate_with_llm(prompt, max_new_tokens=max_tokens, temperature=0.5)

    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        cards = json.loads(raw)
        cards = [c for c in cards if "question" in c and "answer" in c]
    except json.JSONDecodeError:
        cards = []

    return cards


# ──────────────────────────────────────────────────────────────────────────
# Quiz
# ──────────────────────────────────────────────────────────────────────────


def generate_quiz(chunks, difficulty="EZ", num_questions=10, sample_chunks=12):
    step = max(1, len(chunks) // sample_chunks)
    sample = chunks[::step][:sample_chunks]
    context = "\n\n".join(c["text"] for c in sample)

    prompt = f"""
You are a quiz generator for an AI study assistant.

The quiz MUST be based ONLY on the provided study material.

Difficulty:
{QUIZ_MODES[difficulty]}

Generate exactly {num_questions} multiple-choice questions.

Each question must have:
- One question
- Exactly 4 options
- Exactly ONE correct answer
- A short explanation

Return ONLY valid JSON in this format:

{{
    "questions": [
        {{
            "question": "Question here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Why this answer is correct."
        }}
    ]
}}

IMPORTANT:
- correct_answer must be 0, 1, 2, or 3.
- Do not use information outside the study material.
- Every question must have one clearly correct answer.

STUDY MATERIAL:
{context}
"""
    max_tokens = min(8000, 300 * num_questions + 500)
    raw = generate_with_llm(prompt, max_new_tokens=max_tokens, temperature=0.6)

    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        quiz = json.loads(raw)
        questions = quiz.get("questions", [])
        questions = [
            q
            for q in questions
            if "question" in q
            and "options" in q
            and len(q["options"]) == 4
            and "correct_answer" in q
            and q["correct_answer"] in [0, 1, 2, 3]
        ]
    except json.JSONDecodeError:
        questions = []

    return questions


def show_result_image(url, caption):
    try:
        st.image(url, caption=caption, use_container_width=True)
    except Exception:
        st.markdown(f"[View the '{caption}' meme]({url})")


# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────

for key, default in {
    "chunks": None,
    "index": None,
    "llm": None,
    "tts": None,
    "voice_enabled": True,
    "conversation_history": [],
    "flashcards": [],
    "current_card": 0,
    "show_answer": False,
    "quiz_questions": [],
    "quiz_submitted": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — API key, voice toggle, upload & build index
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📚 Study Material")

    _default_api_key = resolve_api_key()
    if _default_api_key:
        api_key = _default_api_key
        st.caption("✅ Groq API key loaded automatically")
    else:
        api_key = st.text_input(
            "Groq API key",
            type="password",
            help="Get one free from console.groq.com/keys. Set GROQ_API_KEY as a "
                 "Streamlit secret or env var and this box won't show up again.",
        )

    voice_enabled = st.checkbox("🔊 Speak answers (Kokoro TTS)", value=True)

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button(
        "Build Study Buddy",
        use_container_width=True,
        disabled=not uploaded_files or not api_key,
    ):
        with st.spinner("Reading, chunking, and embedding your material..."):
            chunks, index, status = build_index(uploaded_files)

        st.session_state.llm = load_llm(api_key)
        if voice_enabled:
            st.session_state.tts = load_tts()
        st.session_state.voice_enabled = voice_enabled

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.conversation_history = []
        st.session_state.flashcards = []
        st.session_state.current_card = 0
        st.session_state.show_answer = False
        st.session_state.quiz_questions = []
        st.session_state.quiz_submitted = False

        if chunks is None:
            st.error(status)
        else:
            st.success(status)

    if st.session_state.chunks is not None:
        st.caption(f"Ready — {len(st.session_state.chunks)} chunks loaded.")

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

st.title("🧠 Study Buddy")
st.write("Chat with your notes, build flashcards, and quiz yourself — all grounded in your own PDFs.")

ready = st.session_state.chunks is not None and st.session_state.index is not None

tab_chat, tab_flashcards, tab_quiz = st.tabs(["💬 Chat", "🗂️ Flashcards", "📝 Quiz"])

# ──────────────────────────────────────────────────────────────────────────
# Chat tab
# ──────────────────────────────────────────────────────────────────────────

with tab_chat:
    if not ready:
        st.info("Upload your PDF(s) and click **Build Study Buddy** in the sidebar.")
    else:
        for message in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.write(message["question"])
            with st.chat_message("assistant"):
                st.write(message["answer"])
                if message.get("audio_path") and os.path.exists(message["audio_path"]):
                    st.audio(message["audio_path"])

        question = st.chat_input("Ask StudyBuddy about your material...")
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = study_buddy_answer(
                        question,
                        st.session_state.conversation_history,
                        st.session_state.chunks,
                        st.session_state.index,
                    )
                st.write(answer)

                audio_path = None
                if st.session_state.voice_enabled and st.session_state.tts is not None:
                    with st.spinner("Generating voice..."):
                        fname = os.path.join(
                            tempfile.gettempdir(),
                            f"studybuddy_{len(st.session_state.conversation_history)}.wav",
                        )
                        audio_path = generate_speech(answer, fname)
                    if audio_path:
                        st.audio(audio_path)

            st.session_state.conversation_history.append(
                {"question": question, "answer": answer, "audio_path": audio_path}
            )

# ──────────────────────────────────────────────────────────────────────────
# Flashcards tab
# ──────────────────────────────────────────────────────────────────────────

with tab_flashcards:
    if not ready:
        st.info("Upload your PDF(s) and click **Build Study Buddy** in the sidebar first.")
    else:
        num_cards = st.slider("Number of flashcards", min_value=4, max_value=20, value=8)

        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                st.session_state.flashcards = generate_flashcards(
                    st.session_state.chunks, num_cards=num_cards
                )
            st.session_state.current_card = 0
            st.session_state.show_answer = False
            if not st.session_state.flashcards:
                st.error("Couldn't parse flashcards from the model's response. Try again.")

        cards = st.session_state.flashcards
        if not cards:
            st.caption("No flashcards yet — click **Generate Flashcards** above.")
        else:
            idx = st.session_state.current_card
            card = cards[idx]

            st.subheader(f"Card {idx + 1} / {len(cards)}")
            if st.session_state.show_answer:
                st.write(card["answer"])
            else:
                st.write(card["question"])

            nav_prev, nav_flip, nav_next = st.columns([1, 2, 1])

            with nav_prev:
                if st.button("⬅️ Previous", use_container_width=True, disabled=idx == 0):
                    st.session_state.current_card -= 1
                    st.session_state.show_answer = False
                    st.rerun()

            with nav_flip:
                flip_label = "🙈 Hide Answer" if st.session_state.show_answer else "🔍 Show Answer"
                if st.button(flip_label, use_container_width=True):
                    st.session_state.show_answer = not st.session_state.show_answer
                    st.rerun()

            with nav_next:
                if st.button("Next ➡️", use_container_width=True, disabled=idx == len(cards) - 1):
                    st.session_state.current_card += 1
                    st.session_state.show_answer = False
                    st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Quiz tab
# ──────────────────────────────────────────────────────────────────────────

with tab_quiz:
    if not ready:
        st.info("Upload your PDF(s) and click **Build Study Buddy** in the sidebar first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox("Difficulty", list(QUIZ_MODES.keys()))
        with col2:
            num_questions = st.slider("Number of questions", min_value=3, max_value=20, value=10)

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                st.session_state.quiz_questions = generate_quiz(
                    st.session_state.chunks,
                    difficulty=difficulty,
                    num_questions=num_questions,
                )
            st.session_state.quiz_submitted = False
            if not st.session_state.quiz_questions:
                st.error("Couldn't parse a quiz from the model's response. Try again.")

        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                user_answers = {}
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    user_answers[i] = st.radio(
                        f"quiz_q_{i}",
                        q["options"],
                        index=None,
                        key=f"quiz_q_{i}",
                        label_visibility="collapsed",
                    )

                submitted = st.form_submit_button("Submit Quiz")

            if submitted:
                score = 0
                answered = 0
                for i, q in enumerate(st.session_state.quiz_questions):
                    selected = user_answers[i]
                    if selected is None:
                        continue
                    answered += 1
                    if selected == q["options"][q["correct_answer"]]:
                        score += 1

                st.session_state.quiz_score = score
                st.session_state.quiz_answered = answered
                st.session_state.quiz_submitted = True
                st.rerun()

        if st.session_state.quiz_submitted:
            score = st.session_state.quiz_score
            answered = st.session_state.quiz_answered
            total = len(st.session_state.quiz_questions)

            st.markdown(f"### Final score: {score}/{answered} (of {total} questions)")

            won = answered > 0 and (score / answered) >= 0.5

            if won:
                st.success(WIN_MESSAGE)
                show_result_image(WIN_IMAGE_URL, "+100000 social credit")
            else:
                st.error(LOSE_MESSAGE)
                show_result_image(LOSE_IMAGE_URL, "-10000 social credit")

            with st.expander("Review answers"):
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    st.write("Correct answer:", q["options"][q["correct_answer"]])
                    st.write("Explanation:", q["explanation"])
                    st.markdown("---")

            if st.button("Take a new quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.rerun()