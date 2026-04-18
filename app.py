from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(
    page_title="Smart FAQ Chatbot (RAG)",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root {
    --bg: #f6f3ec;
    --panel: #fffdf8;
    --text: #1f2937;
    --muted: #6b7280;
    --accent: #215f7c;
    --accent-2: #a85c2f;
    --border: rgba(33, 95, 124, 0.15);
}

body {
    background: linear-gradient(180deg, #f8f4ed 0%, #fffdf8 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 1.2rem 1.4rem;
    border: 1px solid var(--border);
    border-radius: 1.2rem;
    background: radial-gradient(circle at top right, rgba(33,95,124,0.12), transparent 36%),
                linear-gradient(135deg, rgba(33,95,124,0.06), rgba(168,92,47,0.05));
    box-shadow: 0 10px 30px rgba(17,24,39,0.05);
}

.hero h1 {
    margin: 0;
    font-size: 2rem;
}

.hero p {
    margin: 0.4rem 0 0;
    color: var(--muted);
}

.chip {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    margin-right: 0.45rem;
    margin-bottom: 0.45rem;
    border-radius: 999px;
    background: rgba(33,95,124,0.08);
    color: var(--accent);
    font-size: 0.84rem;
}

.source-card {
    border: 1px solid var(--border);
    border-radius: 0.9rem;
    padding: 0.9rem;
    background: rgba(255,255,255,0.78);
    margin-bottom: 0.65rem;
}

.small-muted {
    color: var(--muted);
    font-size: 0.92rem;
}
</style>
"""


SAMPLE_KB = [
    {
        "question": "What are your working hours?",
        "answer": "We are available Monday to Friday from 9:00 AM to 6:00 PM. Support is closed on weekends and public holidays.",
    },
    {
        "question": "What is your refund policy?",
        "answer": "Refunds are available within 14 days of purchase if the service has not been substantially used. Submit the order number and reason to support.",
    },
    {
        "question": "How can I reset my password?",
        "answer": "If you cannot log in, use the password reset link on the sign-in page. For locked accounts, contact support with your registered email address.",
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days depending on location.",
    },
    {
        "question": "Where can I find invoices?",
        "answer": "Invoices are sent automatically after payment. If you need a VAT invoice, include your company details in the billing profile.",
    },
]


def init_state() -> None:
    defaults = {
        "index_ready": False,
        "docs": [],
        "chunks": [],
        "word_vectorizer": None,
        "word_matrix": None,
        "char_vectorizer": None,
        "char_matrix": None,
        "corpus_hash": None,
        "messages": [],
        "conversations": [],
        "active_conversation_id": "current",
        "last_sources": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@dataclass
class Chunk:
    doc_name: str
    text: str
    chunk_id: int
    question: str = ""
    answer: str = ""


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def split_into_chunks(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = sentence
        else:
            chunks.append(sentence[:max_chars])
            current = sentence[max_chars - overlap :]

    if current:
        chunks.append(current)

    refined: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            refined.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = min(len(chunk), start + max_chars)
                refined.append(chunk[start:end])
                if end == len(chunk):
                    break
                start = max(0, end - overlap)
    return [c for c in refined if c.strip()]


def parse_faq_pairs(text: str) -> list[tuple[str, str]]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    lines = [line.rstrip() for line in text.splitlines()]
    pairs: list[tuple[str, str]] = []

    current_question = ""
    current_answer_lines: list[str] = []
    saw_numbered_question = False

    def flush_pair() -> None:
        nonlocal current_question, current_answer_lines, saw_numbered_question
        question = normalize_text(current_question).rstrip(" ?")
        answer = normalize_text(" ".join(current_answer_lines))
        if question and answer:
            pairs.append((question, answer))
        current_question = ""
        current_answer_lines = []
        saw_numbered_question = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_question:
                current_answer_lines.append("")
            continue

        numbered_match = re.match(r"^\d+\.\s*(.+)$", line)
        if numbered_match:
            if current_question:
                flush_pair()
            current_question = numbered_match.group(1).strip()
            saw_numbered_question = True
            continue

        qa_question_match = re.match(r"^(?:Q(?:uestion)?\s*[:\-]\s*)(.+)$", line, flags=re.I)
        if qa_question_match:
            if current_question:
                flush_pair()
            current_question = qa_question_match.group(1).strip()
            saw_numbered_question = False
            continue

        qa_answer_match = re.match(r"^(?:A(?:nswer)?\s*[:\-]\s*)(.+)$", line, flags=re.I)
        if qa_answer_match and current_question:
            current_answer_lines.append(qa_answer_match.group(1).strip())
            continue

        if current_question:
            current_answer_lines.append(line)
            continue

        if not current_question and "?" in line:
            question_part, answer_part = line.split("?", 1)
            current_question = f"{question_part.strip()}?"
            if answer_part.strip():
                current_answer_lines.append(answer_part.strip())
                flush_pair()

    if current_question:
        flush_pair()

    return pairs


def read_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_uploaded_file(uploaded_file) -> str:
    suffix = uploaded_file.name.lower().split(".")[-1]
    data = uploaded_file.getvalue()
    if suffix == "pdf":
        return read_pdf_bytes(data)
    if suffix in {"txt", "md", "csv"}:
        return data.decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def parse_csv_text(text: str) -> str:
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception:
        return text
    if not rows:
        return text
    return "\n".join(", ".join(row) for row in rows)


def make_chunks_from_sources(sources: list[tuple[str, str]], max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_name, text in sources:
        if doc_name.lower().endswith(".csv"):
            text = parse_csv_text(text)
        pairs = parse_faq_pairs(text)
        if pairs:
            for idx, (question, answer) in enumerate(pairs):
                combined = f"Question: {question}\nAnswer: {answer}"
                chunks.append(
                    Chunk(
                        doc_name=doc_name,
                        text=combined,
                        chunk_id=idx,
                        question=question,
                        answer=answer,
                    )
                )
            continue

        for idx, chunk_text in enumerate(split_into_chunks(text, max_chars=max_chars)):
            chunks.append(Chunk(doc_name=doc_name, text=chunk_text, chunk_id=idx))
    return chunks


def corpus_fingerprint(sources: Iterable[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name, text in sources:
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def build_index(chunks: list[Chunk]):
    texts = [chunk.question or chunk.text for chunk in chunks]
    if not texts:
        return None, None, None, None
    word_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=7000,
    )
    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)
    return word_vectorizer, word_matrix, char_vectorizer, char_matrix


def extract_keywords(query: str, limit: int = 5) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", query.lower())
    stop = {
        "the",
        "and",
        "or",
        "for",
        "with",
        "what",
        "when",
        "how",
        "where",
        "why",
        "can",
        "you",
        "tell",
        "about",
        "please",
        "i",
        "need",
        "to",
        "a",
        "an",
        "is",
        "are",
        "do",
        "does",
        "my",
        "we",
        "our",
    }
    words = [t for t in tokens if len(t) > 5 and t not in stop]
    return [word for word, _ in Counter(words).most_common(limit)]


def rank_chunks(
    query: str,
    word_vectorizer,
    word_matrix,
    char_vectorizer,
    char_matrix,
    chunks: list[Chunk],
    top_k: int = 4,
):
    if (
        not chunks
        or word_vectorizer is None
        or word_matrix is None
        or char_vectorizer is None
        or char_matrix is None
    ):
        return []

    q_word = word_vectorizer.transform([query])
    q_char = char_vectorizer.transform([query])
    word_scores = (word_matrix @ q_word.T).toarray().ravel()
    char_scores = (char_matrix @ q_char.T).toarray().ravel()
    scores = (0.35 * word_scores) + (0.65 * char_scores)

    order = np.argsort(scores)[::-1]
    results = []
    for idx in order[:top_k]:
        score = float(scores[idx])
        if score <= 0.08:
            continue
        chunk = chunks[idx]
        results.append((score, chunk))
    return results


def build_answer(query: str, retrieved: list[tuple[float, Chunk]]) -> str:
    if not retrieved:
        return (
            "I could not find a confident answer in the uploaded FAQ files. "
            "Try rephrasing the question or upload a more relevant document."
        )

    best_score, best_chunk = retrieved[0]

    if best_score < 0.55:
        suggestions = []
        for _, chunk in retrieved[:3]:
            label = chunk.question or chunk.text.splitlines()[0]
            suggestions.append(label.strip())
        suggestion_text = "; ".join(suggestions) if suggestions else "a more specific FAQ entry"
        return (
            "I could not find a confident answer in the FAQ.\n\n"
            f"Closest FAQ questions: {suggestion_text}"
        )

    if best_chunk.answer:
        return best_chunk.answer

    keywords = extract_keywords(query)
    lead = "Here is the most relevant information I found:"
    if keywords:
        lead = f"Here is the most relevant information about {', '.join(keywords[:3])}:"

    bullets = []
    for score, chunk in retrieved[:3]:
        snippet = chunk.text
        if len(snippet) > 180:
            snippet = snippet[:177].rstrip() + "..."
        bullets.append(f"- {snippet} [{chunk.doc_name}]")

    return f"{lead}\n" + "\n".join(bullets) + "\n\nIf you want, I can keep searching across the uploaded FAQ set."


def format_sources(retrieved: list[tuple[float, Chunk]]) -> list[dict[str, str]]:
    items = []
    for score, chunk in retrieved[:4]:
        items.append(
            {
                "document": chunk.doc_name,
                "question": chunk.question or "",
                "chunk": str(chunk.chunk_id + 1),
                "score": f"{score:.3f}",
                "answer": chunk.answer[:260] if chunk.answer else chunk.text[:260],
            }
        )
    return items


def index_sources(sources: list[tuple[str, str]], max_chars: int, progress=None):
    chunks = make_chunks_from_sources(sources, max_chars=max_chars)
    if progress is not None:
        progress.progress(30, text="Chunking FAQ content...")
    word_vectorizer, word_matrix, char_vectorizer, char_matrix = build_index(chunks)
    if progress is not None:
        progress.progress(85, text="Building semantic index...")
    return chunks, word_vectorizer, word_matrix, char_vectorizer, char_matrix


def start_new_chat() -> None:
    if st.session_state.messages:
        st.session_state.conversations.insert(
            0,
            {
                "title": st.session_state.messages[0]["content"][:48],
                "messages": st.session_state.messages.copy(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        )
    st.session_state.messages = []
    st.session_state.active_conversation_id = "current"


def render_sidebar() -> None:
    st.sidebar.markdown("## Controls")
    st.sidebar.caption("Upload FAQ files, then ask questions in natural language.")

    max_chars = st.sidebar.slider("Chunk size", 300, 1200, 800, 50)
    top_k = st.sidebar.slider("Top matches", 1, 6, 3, 1)
    use_sample = st.sidebar.toggle("Use sample FAQ", value=True)

    uploaded_files = st.sidebar.file_uploader(
        "Upload FAQ documents",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Upload one or more knowledge base files.",
    )

    if st.sidebar.button("Build / Refresh index", use_container_width=True):
        sources: list[tuple[str, str]] = []
        if use_sample:
            for idx, item in enumerate(SAMPLE_KB, 1):
                sources.append((f"sample_faq_{idx}.txt", f"{idx}. {item['question']}\n{item['answer']}"))
        if uploaded_files:
            for file in uploaded_files:
                sources.append((file.name, read_uploaded_file(file)))

        if not sources:
            st.sidebar.warning("Add at least one FAQ file or enable the sample FAQ.")
        else:
            fingerprint = corpus_fingerprint(sources)
            if fingerprint == st.session_state.corpus_hash and st.session_state.index_ready:
                st.sidebar.success("Index is already up to date.")
            else:
                progress = st.sidebar.progress(0, text="Preparing documents...")
                chunks, word_vectorizer, word_matrix, char_vectorizer, char_matrix = index_sources(
                    sources, max_chars=max_chars, progress=progress
                )
                progress.progress(100, text="Index ready")

                st.session_state.docs = sources
                st.session_state.chunks = chunks
                st.session_state.word_vectorizer = word_vectorizer
                st.session_state.word_matrix = word_matrix
                st.session_state.char_vectorizer = char_vectorizer
                st.session_state.char_matrix = char_matrix
                st.session_state.corpus_hash = fingerprint
                st.session_state.index_ready = True
                st.session_state.last_sources = []
                st.sidebar.success(f"Indexed {len(sources)} documents and {len(chunks)} chunks.")

    st.sidebar.divider()
    st.sidebar.markdown("### Conversation history")
    if st.session_state.conversations:
        options = ["Current chat"] + [
            f"{item['created_at']} - {item['title']}" for item in st.session_state.conversations
        ]
        selected = st.sidebar.selectbox("Load conversation", options, index=0)
        if selected != "Current chat":
            idx = options.index(selected) - 1
            st.session_state.messages = st.session_state.conversations[idx]["messages"].copy()
            st.sidebar.info("Loaded saved conversation into the main chat.")
    else:
        st.sidebar.caption("No saved chats yet.")

    if st.sidebar.button("New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### Status")
    st.sidebar.write("Index ready" if st.session_state.index_ready else "Index not built yet")
    st.sidebar.write(f"Documents: {len(st.session_state.docs)}")
    st.sidebar.write(f"Messages: {len(st.session_state.messages)}")

    return top_k, max_chars


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>Smart FAQ Chatbot (RAG)</h1>
            <p>Upload company FAQ documents, index them semantically, and answer user questions with chat history and source references.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_k, max_chars = render_sidebar()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Indexed docs", len(st.session_state.docs))
    col_b.metric("Chat turns", len(st.session_state.messages))
    col_c.metric("Chunks", len(st.session_state.chunks))

    st.caption("Tip: keep the sample FAQ enabled for a ready-to-demo chatbot, then replace it with your own documents.")

    if not st.session_state.index_ready:
        st.info("Build the index from the sidebar first. The sample FAQ is enough to test the full chat flow.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources used"):
                    st.dataframe(message["sources"], use_container_width=True, hide_index=True)

    user_query = st.chat_input("Ask a question about the FAQ...")

    if user_query:
        if (
            not st.session_state.index_ready
            or st.session_state.word_vectorizer is None
            or st.session_state.word_matrix is None
            or st.session_state.char_vectorizer is None
            or st.session_state.char_matrix is None
        ):
            st.warning("Please build the FAQ index first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        retrieved = rank_chunks(
            user_query,
            st.session_state.word_vectorizer,
            st.session_state.word_matrix,
            st.session_state.char_vectorizer,
            st.session_state.char_matrix,
            st.session_state.chunks,
            top_k=top_k,
        )
        answer = build_answer(user_query, retrieved)
        sources = format_sources(retrieved)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("Sources used"):
                    st.dataframe(sources, use_container_width=True, hide_index=True)

    if st.session_state.messages:
        st.sidebar.divider()
        st.sidebar.markdown("### Recent answer")
        last_assistant = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            None,
        )
        if last_assistant:
            st.sidebar.write(last_assistant["content"][:280])


if __name__ == "__main__":
    main()
