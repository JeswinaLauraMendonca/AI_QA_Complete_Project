import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .metric-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD QA ENGINE
# ============================================================

@st.cache_resource
def load_qa_engine():

    from backend.qa_engine import QAEngine

    return QAEngine()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI-Based Question Answering System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Retrieval-based Question Answering using BM25, fine-tuned
    All-MiniLM-L6-v2, RNN/GRU, and Hybrid Retrieval
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Query Settings")

    method = st.selectbox(
        "Select Question Answering Method",
        options=[
            "Hybrid",
            "BM25",
            "All-MiniLM-L6-v2",
            "RNN/GRU"
        ]
    )

    method_map = {
        "Hybrid": "hybrid",
        "BM25": "bm25",
        "All-MiniLM-L6-v2": "minilm",
        "RNN/GRU": "rnn"
    }

    selected_method = method_map[method]

    top_k = st.slider(
        "Number of Retrieved Sources",
        min_value=1,
        max_value=5,
        value=3
    )

    st.divider()

    st.subheader("Methods")

    st.write(
        "**BM25**\n"
        "Keyword-based probabilistic retrieval."
    )

    st.write(
        "**All-MiniLM-L6-v2**\n"
        "Semantic retrieval using sentence embeddings."
    )

    st.write(
        "**RNN/GRU**\n"
        "Neural candidate reranking using a GRU model."
    )

    st.write(
        "**Hybrid**\n"
        "Combines BM25, MiniLM, and RNN/GRU scores."
    )

    st.divider()

    st.caption(
        "AI Question Answering System\n"
        "BE CSE-AIML Project"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask a Question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What is the capital of Pakistan?"
    ),
    height=100
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Get Answer",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question before clicking Get Answer."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                engine = load_qa_engine()

                result = engine.query(
                    question=question,
                    method=selected_method,
                    top_k=top_k
                )

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )

                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("💡 Answer")

        answer = result.get(
            "answer",
            ""
        )

        if answer:

            st.markdown(
                f"""
                <div class="answer-box">
                <strong>{answer}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "No answer could be extracted from the retrieved context."
            )


        # ====================================================
        # METRICS
        # ====================================================

        st.subheader("📊 Query Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Method",
                method
            )

        with col2:

            st.metric(
                "Response Latency",
                f"{result['latency']:.4f} s"
            )

        with col3:

            st.metric(
                "Sources Retrieved",
                len(result["results"])
            )


        # ====================================================
        # RETRIEVED SOURCES
        # ====================================================

        st.subheader("📚 Retrieved Sources")

        retrieved_results = result.get(
            "results",
            []
        )

        if not retrieved_results:

            st.info(
                "No retrieval results available."
            )

        else:

            for i, item in enumerate(
                retrieved_results,
                start=1
            ):

                with st.expander(
                    f"Source {i} — Score: {item['score']:.4f}"
                ):

                    st.write(
                        item["context"]
                    )

                    st.caption(
                        f"Context index: {item['index']}"
                    )


# ============================================================
# INFORMATION SECTION
# ============================================================

st.divider()

with st.expander("ℹ️ About this system"):

    st.write(
        """
        This system implements a Retrieval-Augmented Question
        Answering architecture.

        The system uses multiple retrieval approaches:

        • BM25 for lexical retrieval

        • Fine-tuned All-MiniLM-L6-v2 for semantic retrieval

        • RNN/GRU for neural candidate reranking

        • Hybrid retrieval for combining the individual scores

        A local DistilBERT extractive reader is used to extract
        the final answer from the highest-ranked context.
        """
    )
    