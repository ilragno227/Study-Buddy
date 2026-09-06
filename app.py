"""
Study Buddy — Streamlit App
============================
A RAG-powered study assistant with three tabs: Chat, Flashcards, and Quiz.
Upload one or more study PDFs, build the vector index, then chat with the
material, generate flashcards, or take a generated multiple-choice quiz.

Run locally:
    streamlit run App.py
"""

import json
import re

import faiss
import numpy as np
import streamlit as st
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ──────────────────────────────────────────────────────────────────────────
# Config & constants
# ──────────────────────────────────────────────────────────────────────────

GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Replace these with direct .gif/.png links (right click the Tenor GIF ->
# "Copy image address") — Tenor's tenor.com/view/... page links are not
# directly embeddable as an <img> source.
WIN_IMAGE_URL = "https://tenor.com/view/social-credit-credit-social-уважение-плюс-уважение-gif-1626328442317885176"
LOSE_IMAGE_URL = "https://tenor.com/view/nalog-gif-25906765"

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
# Page config & styling (matches the Study Buddy logo: olive green + cream)
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Study Buddy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --sb-bg: #eeeeea;
    --sb-olive: #7c8a3e;
    --sb-olive-dark: #5a6530;
    --sb-olive-darker: #454e24;
    --sb-black: #232318;
    --sb-cream: #f5f5ee;
}

.stApp {
    background-color: var(--sb-bg);
}

h1, h2, h3 {
    color: var(--sb-olive-darker) !important;
    font-weight: 800 !important;
}

/* Title banner */
.sb-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}
.sb-title h1 {
    margin: 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.sb-subtitle {
    color: var(--sb-olive-dark);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Buttons */
.stButton>button, .stFormSubmitButton>button {
    background-color: var(--sb-olive);
    color: white;
    border: 2px solid var(--sb-olive-darker);
    border-radius: 10px;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    transition: background-color 0.15s ease-in-out;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    background-color: var(--sb-olive-dark);
    border-color: var(--sb-black);
    color: white;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: var(--sb-cream);
    border-radius: 10px 10px 0 0;
    border: 2px solid var(--sb-olive);
    border-bottom: none;
    padding: 8px 18px;
    font-weight: 700;
    color: var(--sb-olive-darker);
}
.stTabs [aria-selected="true"] {
    background-color: var(--sb-olive) !important;
    color: white !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--sb-cream);
    border-right: 3px solid var(--sb-olive);
}

/* Cards (flashcards / quiz result) */
.sb-card {
    background-color: var(--sb-cream);
    border: 2px solid var(--sb-olive);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
}
.sb-card .sb-q {
    font-weight: 700;
    color: var(--sb-olive-darker);
    margin-bottom: 0.4rem;
}
.sb-card .sb-a {
    color: var(--sb-black);
}

/* Result banner */
.sb-result-win {
    background-color: var(--sb-olive);
    color: white;
    text-align: center;
    padding: 1.5rem;
    border-radius: 14px;
    font-size: 1.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.sb-result-lose {
    background-color: var(--sb-black);
    color: var(--sb-cream);
    text-align: center;
    padding: 1.5rem;
    border-radius: 14px;
    font-size: 1.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ──────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_resource(show_spinner="Loading Qwen LLM (this can take a while)...")
def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    return tokenizer, model, device


def generate_with_llm(prompt, max_new_tokens=1024, temperature=0.7):
    tokenizer, model, device = load_llm()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


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
    raw = generate_with_llm(prompt, max_new_tokens=1500, temperature=0.5)

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
    raw = generate_with_llm(prompt, max_new_tokens=2000, temperature=0.6)

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
    """Try to render the meme inline; fall back to a link if the URL
    isn't a direct image (e.g. a tenor.com/view/... page link)."""
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
    "conversation_history": [],
    "flashcards": [],
    "quiz_questions": [],
    "quiz_submitted": False,
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

    if st.button("Build Study Buddy", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading, chunking, and embedding your material..."):
                chunks, index = build_index(uploaded_files)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.conversation_history = []
                st.session_state.flashcards = []
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.chunks is not None:
        st.caption(f"Ready — {len(st.session_state.chunks)} chunks loaded.")

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="sb-title"><h1>🧠 Study Buddy</h1></div>
    <div class="sb-subtitle">Chat with your notes, build flashcards, and quiz yourself — all grounded in your own PDFs.</div>
    """,
    unsafe_allow_html=True,
)

ready = st.session_state.chunks is not None and st.session_state.index is not None

tab_chat, tab_flashcards, tab_quiz = st.tabs(["💬 Chat", "🗂️ Flashcards", "📝 Quiz"])

# ──────────────────────────────────────────────────────────────────────────
# Chat tab
# ──────────────────────────────────────────────────────────────────────────

with tab_chat:
    if not ready:
        st.info("Upload your PDF(s) and click **Build Study Buddy** in the sidebar to start chatting.")
    else:
        for message in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.write(message["question"])
            with st.chat_message("assistant"):
                st.write(message["answer"])

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
                st.error("Couldn't parse flashcards from the model's response. Try again.")

        if st.session_state.flashcards:
            for i, card in enumerate(st.session_state.flashcards, start=1):
                st.markdown(
                    f"""
                    <div class="sb-card">
                        <div class="sb-q">Card {i}: {card['question']}</div>
                        <div class="sb-a">{card['answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
                    st.markdown("&nbsp;", unsafe_allow_html=True)

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
                st.markdown(f'<div class="sb-result-win">{WIN_MESSAGE}</div>', unsafe_allow_html=True)
                show_result_image(WIN_IMAGE_URL, "+100000 social credit")
            else:
                st.markdown(f'<div class="sb-result-lose">{LOSE_MESSAGE}</div>', unsafe_allow_html=True)
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
