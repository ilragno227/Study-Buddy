"""
Study Buddy — Streamlit App
============================
A RAG-powered study assistant with three tabs: Chat, Flashcards, and Quiz.
Upload one or more study PDFs, build the vector index, then chat with the
material, generate flashcards, or take a generated multiple-choice quiz.
Chat answers can optionally be read aloud with a local Kokoro TTS voice.

Run locally:
    streamlit run app.py
"""

import io
import json
import os
import re
import time

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Kokoro TTS is optional: if it (or one of its own dependencies, e.g. the
# espeak-ng system package) isn't available, the rest of the app must keep
# working — voice just quietly disables itself.
try:
    from kokoro import KPipeline
    import soundfile as sf

    TTS_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on deployment environment
    KPipeline = None
    sf = None
    TTS_IMPORT_ERROR = str(e)

TTS_AVAILABLE = TTS_IMPORT_ERROR is None

# ──────────────────────────────────────────────────────────────────────────
# Config & constants
# ──────────────────────────────────────────────────────────────────────────

# Pinned to a specific stable version rather than the "-latest" alias —
# aliases are convenient but have been less reliable in practice. Update
# this manually if Google deprecates the version (check
# https://ai.google.dev/gemini-api/docs/models for current options).
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini 3.x models "think" before answering by default, and — this is the
# part that isn't obvious from the docs — those hidden thinking tokens are
# deducted from the SAME max_output_tokens budget as the visible answer.
# With a low max_output_tokens (the previous version used 1024-2000) the
# model can spend the entire budget thinking and leave nothing for the
# actual reply, which is exactly what caused answers/flashcards/quizzes to
# come back empty or cut off mid-sentence. The fix is two-part: keep
# thinking effort low (we don't need deep reasoning for grounded Q&A over a
# student's own notes) AND give each task a generous token ceiling.
THINKING_LEVEL = genai_types.ThinkingLevel.LOW

MAX_TOKENS_CHAT = 4096
MAX_TOKENS_FLASHCARDS = 4096
MAX_TOKENS_QUIZ = 6144

# Optional local images shown on the quiz results screen. The previous
# version hot-linked tenor.com "view" page URLs, which are HTML pages, not
# image files — st.image can't render those, which is why the images never
# loaded. Direct-linking third-party meme CDNs is also fragile long-term
# (links rot). If you want a custom image, drop a file at one of these
# paths in your repo; otherwise the app falls back to a built-in Streamlit
# celebration effect, which always works with zero setup.
WIN_IMAGE_PATH = "assets/win.gif"
LOSE_IMAGE_PATH = "assets/lose.gif"

WIN_MESSAGE = "You Win, gg wp"
LOSE_MESSAGE = "You Lose, train harder twin!"

# Kokoro voice settings (matches the original notebook prototype).
TTS_VOICE = "af_heart"
TTS_SPEED = 1.05
TTS_LANG_CODE = "a"  # American English
TTS_SAMPLE_RATE = 24000

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

SYSTEM_PROMPT = """
IDENTITY

You are StudyBuddy, a sharp and encouraging AI study coach. A student
has handed you a PDF they're trying to learn, and your job is to make
it click for them - not recite it back.

GROUNDING RULES

1. Answer using only the retrieved PDF context you're given. Do not add
   outside facts, even ones you're confident are true.
2. If the context doesn't contain the answer, say so plainly - for
   example: "I don't see that covered in this material." Never imply the
   material said something it didn't.
3. You may reference earlier turns in the conversation for continuity,
   but the retrieved context is always the source of truth for facts.

HOW YOU TEACH

- Lead with plain language, then introduce the technical term once the
  idea already makes sense.
- Use a short real-world analogy only when it genuinely clarifies
  something.
- Keep answers tight: a few short paragraphs by default. Only go longer
  if the student explicitly asks for more depth.
- Never say "as an AI" or reference being a language model.
"""

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
# No custom CSS / color overrides — using Streamlit's default theme as
# requested. Native components (st.container(border=True), st.success,
# st.error, etc.) already look consistent and adapt to light/dark mode for
# free, which the old hand-rolled HTML+CSS cards didn't.

st.set_page_config(
    page_title="Study Buddy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ──────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_gemini_client():
    # Reads the key from Streamlit secrets first, then falls back to an
    # environment variable — set one of these before running the app.
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        st.error(
            "No Gemini API key found. Add GEMINI_API_KEY to your Streamlit "
            "secrets (Settings → Secrets) or as an environment variable."
        )
        st.stop()
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner="Loading Kokoro voice model (first run only)...")
def load_tts_pipeline():
    # Raises on failure; callers catch this rather than us swallowing it
    # here, so a transient failure (e.g. HF Hub hiccup on first download)
    # can be retried instead of being cached as a permanent None.
    return KPipeline(lang_code=TTS_LANG_CODE, device="cpu")


def get_tts_pipeline_safe():
    """Returns (pipeline, error_message). Never raises."""
    if not TTS_AVAILABLE:
        return None, (
            "Kokoro isn't installed in this environment "
            f"({TTS_IMPORT_ERROR}). Check requirements.txt and packages.txt."
        )
    try:
        return load_tts_pipeline(), None
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────────────────────────────────────
# Gemini calls
# ──────────────────────────────────────────────────────────────────────────


def _build_config(max_output_tokens, temperature):
    return genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=genai_types.ThinkingConfig(thinking_level=THINKING_LEVEL),
    )


def _finish_reason_name(response):
    if response.candidates:
        reason = response.candidates[0].finish_reason
        return reason.name if reason is not None else None
    return None


def call_gemini(prompt, max_output_tokens=2048, temperature=0.7, retries=2):
    """Non-streaming call. Returns (text, truncated) where truncated is True
    if the model hit MAX_TOKENS (thinking + answer together ran out of
    room) — callers use this to retry with a bigger budget instead of
    silently returning an empty string."""
    client = load_gemini_client()
    config = _build_config(max_output_tokens, temperature)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            truncated = _finish_reason_name(response) == "MAX_TOKENS"
            return text, truncated

        except genai_errors.ServerError as e:
            # Gemini's own servers returned a 5xx — usually transient.
            # Back off briefly and retry before giving up.
            last_error = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue

        except genai_errors.ClientError as e:
            # 4xx — retrying won't help (bad key, bad request, quota, etc).
            st.error(f"Gemini rejected the request: {e}")
            st.stop()

    st.error(
        f"Gemini's servers returned an error after {retries + 1} attempts: "
        f"{last_error}\n\nThis is usually transient — try again in a moment. "
        "If it persists, check https://status.cloud.google.com for outages."
    )
    st.stop()


def call_gemini_with_headroom(prompt, max_output_tokens, temperature=0.7):
    """Calls Gemini and, if the response was cut off by MAX_TOKENS with
    little or nothing to show for it, retries once with double the token
    budget. This is the safety net for the thinking-tokens-eat-the-budget
    problem described above."""
    text, truncated = call_gemini(prompt, max_output_tokens, temperature)
    if truncated and len(text) < 20:
        text, truncated = call_gemini(
            prompt, min(max_output_tokens * 2, 16000), temperature
        )
    return text, truncated


def stream_chat_response(prompt, max_output_tokens, temperature, result_holder):
    """Generator for st.write_stream. Yields visible text chunks as they
    arrive; records the finish reason into result_holder once the stream
    ends so the caller can warn on truncation."""
    client = load_gemini_client()
    config = _build_config(max_output_tokens, temperature)
    try:
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL, contents=prompt, config=config
        )
        last_chunk = None
        for chunk in stream:
            last_chunk = chunk
            piece = chunk.text
            if piece:
                yield piece
        result_holder["truncated"] = (
            last_chunk is not None and _finish_reason_name(last_chunk) == "MAX_TOKENS"
        )
    except genai_errors.APIError as e:
        result_holder["error"] = str(e)
        yield f"\n\n⚠️ Gemini error: {e}"


# ──────────────────────────────────────────────────────────────────────────
# PDF processing pipeline
# ──────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(file_obj, source_name):
    reader = PdfReader(file_obj)
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
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "page": page["page"], "source": page["source"]})
            start += chunk_size - overlap
    return chunks


def build_index(uploaded_files):
    embedding_model = load_embedding_model()

    all_pages = []
    for f in uploaded_files:
        pages = extract_text_from_pdf(f, f.name)
        for p in pages:
            p["text"] = clean_text(p["text"])
        all_pages.extend(pages)

    chunks = create_chunks(all_pages)

    if not chunks:
        st.error(
            "Couldn't extract any text from the uploaded PDF(s). If these "
            "are scanned/image-only pages, they'll need OCR before Study "
            "Buddy can index them."
        )
        return None, None

    embeddings = embedding_model.encode(
        [c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index


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


def build_chat_prompt(question, conversation_history, chunks, index, top_k=5):
    results = retrieve_relevant_chunks(question, chunks, index, top_k=top_k)
    context = build_context(results)

    history = ""
    for message in conversation_history:
        history += f"\nStudent: {message['question']}\nStudyBuddy: {message['answer']}\n"

    return f"""
{SYSTEM_PROMPT}

PREVIOUS CONVERSATION:
{history}

RETRIEVED CONTEXT:
{context}

CURRENT STUDENT QUESTION:
{question}

Answer the student naturally while staying grounded in the retrieved context.
"""


# ──────────────────────────────────────────────────────────────────────────
# Robust JSON extraction (handles responses truncated mid-array/object)
# ──────────────────────────────────────────────────────────────────────────


def _strip_fences(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return raw


def _extract_json_array(raw):
    """Parses a JSON array from raw text, repairing a truncated tail by
    dropping back to the last complete element if needed."""
    raw = _strip_fences(raw)
    start = raw.find("[")
    if start == -1:
        return None
    raw = raw[start:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    last_close = raw.rfind("}")
    if last_close == -1:
        return None
    candidate = raw[: last_close + 1] + "]"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_json_object_array(raw, array_key):
    """Same idea as _extract_json_array but for a top-level object like
    {"questions": [...]} that may have been cut off mid-array."""
    raw = _strip_fences(raw)
    start = raw.find("{")
    if start == -1:
        return None
    raw = raw[start:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    arr_start = raw.find("[")
    last_close = raw.rfind("}")
    if arr_start == -1 or last_close == -1 or last_close <= arr_start:
        return None
    candidate_array = raw[arr_start : last_close + 1] + "]"
    try:
        return {array_key: json.loads(candidate_array)}
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────────────────────────────────
# Flashcards
# ──────────────────────────────────────────────────────────────────────────


def _sample_context(chunks, sample_chunks=12):
    step = max(1, len(chunks) // sample_chunks)
    sample = chunks[::step][:sample_chunks]
    return "\n\n".join(c["text"] for c in sample)


def generate_flashcards(chunks, num_cards=8, sample_chunks=12):
    context = _sample_context(chunks, sample_chunks)

    for attempt_cards in (num_cards, max(4, num_cards // 2)):
        prompt = FLASHCARD_PROMPT_TEMPLATE.format(num_cards=attempt_cards, context=context)
        raw, _ = call_gemini_with_headroom(
            prompt, max_output_tokens=MAX_TOKENS_FLASHCARDS, temperature=0.5
        )
        cards = _extract_json_array(raw)
        if cards:
            return [c for c in cards if "question" in c and "answer" in c]

    return []


# ──────────────────────────────────────────────────────────────────────────
# Quiz
# ──────────────────────────────────────────────────────────────────────────


def _build_quiz_prompt(context, difficulty, num_questions):
    return f"""
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


def generate_quiz(chunks, difficulty="EZ", num_questions=10, sample_chunks=12):
    context = _sample_context(chunks, sample_chunks)

    for attempt_questions in (num_questions, max(3, num_questions // 2)):
        prompt = _build_quiz_prompt(context, difficulty, attempt_questions)
        raw, _ = call_gemini_with_headroom(
            prompt, max_output_tokens=MAX_TOKENS_QUIZ, temperature=0.6
        )
        quiz = _extract_json_object_array(raw, "questions")
        if not quiz:
            continue

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
        if questions:
            return questions

    return []


def show_result_banner(won):
    if won:
        st.success(f"🎉 **{WIN_MESSAGE}**")
        if os.path.exists(WIN_IMAGE_PATH):
            st.image(WIN_IMAGE_PATH, width="stretch")
        else:
            st.balloons()
    else:
        st.error(f"💀 **{LOSE_MESSAGE}**")
        if os.path.exists(LOSE_IMAGE_PATH):
            st.image(LOSE_IMAGE_PATH, width="stretch")
        else:
            st.snow()


# ──────────────────────────────────────────────────────────────────────────
# Text-to-speech (Kokoro)
# ──────────────────────────────────────────────────────────────────────────


def clean_for_speech(text):
    text = re.sub(r"[*_#>`]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_speech_bytes(text):
    """Returns WAV bytes, or (None, error_message) on failure. Kept in
    memory (no temp files) since Streamlit Cloud's filesystem is ephemeral
    and shared across a session's reruns."""
    pipeline, err = get_tts_pipeline_safe()
    if pipeline is None:
        return None, err

    cleaned = clean_for_speech(text)
    if not cleaned:
        return None, "Nothing to read aloud."

    try:
        audio_parts = []
        for _, _, audio in pipeline(
            cleaned,
            voice=TTS_VOICE,
            speed=TTS_SPEED,
            split_pattern=r"(?<=[.!?])\s+",
        ):
            if audio is not None:
                audio_parts.append(np.asarray(audio, dtype=np.float32))

        if not audio_parts:
            return None, "Kokoro produced no audio for this text."

        audio = np.concatenate(audio_parts)
        buffer = io.BytesIO()
        sf.write(buffer, audio, TTS_SAMPLE_RATE, format="WAV")
        return buffer.getvalue(), None
    except Exception as e:
        return None, str(e)


def render_listen_button(text, key):
    """A small on-demand 'listen' control. Generating audio for every past
    message eagerly would be slow and memory-heavy, so it's done lazily on
    click and cached in session_state per message key."""
    audio_key = f"audio_{key}"
    if st.button("🔊 Listen", key=f"btn_{key}"):
        with st.spinner("Generating voice..."):
            audio_bytes, err = generate_speech_bytes(text)
        if err:
            st.warning(f"Voice unavailable: {err}")
        else:
            st.session_state[audio_key] = audio_bytes

    if audio_key in st.session_state:
        st.audio(st.session_state[audio_key], format="audio/wav")


# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────

for key, default in {
    "chunks": None,
    "index": None,
    "conversation_history": [],
    "flashcards": [],
    "quiz_questions": [],
    "quiz_submitted": False,
    "auto_voice": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — upload & build index
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📚 Study Material")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build Study Buddy", width="stretch"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading, chunking, and embedding your material..."):
                chunks, index = build_index(uploaded_files)
            if chunks is not None:
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.conversation_history = []
                st.session_state.flashcards = []
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.chunks is not None:
        st.caption(f"Ready — {len(st.session_state.chunks)} chunks loaded.")

    st.markdown("### 🔊 Voice")
    if TTS_AVAILABLE:
        st.session_state.auto_voice = st.checkbox(
            "Read new answers aloud", value=st.session_state.auto_voice
        )
    else:
        st.caption(f"Voice unavailable: {TTS_IMPORT_ERROR}")

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

st.title("🧠 Study Buddy")
st.caption("Chat with your notes, build flashcards, and quiz yourself — all grounded in your own PDFs.")

ready = st.session_state.chunks is not None and st.session_state.index is not None

tab_chat, tab_flashcards, tab_quiz = st.tabs(["💬 Chat", "🗂️ Flashcards", "📝 Quiz"])

# ──────────────────────────────────────────────────────────────────────────
# Chat tab
# ──────────────────────────────────────────────────────────────────────────

with tab_chat:
    if not ready:
        st.info("Upload your PDF(s) and click **Build Study Buddy** in the sidebar to start chatting.")
    else:
        for i, message in enumerate(st.session_state.conversation_history):
            with st.chat_message("user"):
                st.write(message["question"])
            with st.chat_message("assistant"):
                st.write(message["answer"])
                if TTS_AVAILABLE:
                    render_listen_button(message["answer"], key=f"chat_{i}")

        question = st.chat_input("Ask StudyBuddy about your material...")
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                prompt = build_chat_prompt(
                    question,
                    st.session_state.conversation_history,
                    st.session_state.chunks,
                    st.session_state.index,
                )
                result_holder = {}
                answer = st.write_stream(
                    stream_chat_response(
                        prompt, MAX_TOKENS_CHAT, temperature=0.7, result_holder=result_holder
                    )
                )
                if result_holder.get("truncated"):
                    st.caption(
                        "⚠️ This answer may have been cut short. Try asking a "
                        "more focused follow-up question."
                    )

                if TTS_AVAILABLE and st.session_state.auto_voice and answer:
                    with st.spinner("Generating voice..."):
                        audio_bytes, err = generate_speech_bytes(answer)
                    if err:
                        st.warning(f"Voice unavailable: {err}")
                    else:
                        st.audio(audio_bytes, format="audio/wav", autoplay=True)

            st.session_state.conversation_history.append(
                {"question": question, "answer": answer}
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
            if not st.session_state.flashcards:
                st.error(
                    "Couldn't get usable flashcards back from Gemini. Try again, "
                    "or generate fewer cards at once."
                )

        if st.session_state.flashcards:
            for i, card in enumerate(st.session_state.flashcards, start=1):
                with st.container(border=True):
                    st.markdown(f"**Card {i}: {card['question']}**")
                    st.write(card["answer"])

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
                st.error(
                    "Couldn't get a usable quiz back from Gemini. Try again, "
                    "or generate fewer questions at once."
                )

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

            st.metric("Final score", f"{score}/{answered}", help=f"Out of {total} questions total")

            won = answered > 0 and (score / answered) >= 0.5
            show_result_banner(won)

            with st.expander("Review answers"):
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    st.write("Correct answer:", q["options"][q["correct_answer"]])
                    st.write("Explanation:", q["explanation"])
                    st.divider()

            if st.button("Take a new quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.rerun()
