import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "MeetingTranscriptAnalyzer.ipynb"
TRANSCRIPT_PATH = PROJECT_DIR / "Meeting_Transcript.txt"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

load_dotenv(PROJECT_DIR / ".env")


@st.cache_resource(show_spinner=False, max_entries=1)
def load_workflow_definitions() -> dict[str, Any]:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Workflow notebook not found: {NOTEBOOK_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    workflow_module = ModuleType("_ai_meeting_notes_workflow")
    sys.modules[workflow_module.__name__] = workflow_module
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.get("cell_type") == "code" and "runtime" in tags:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            exec(compile(source, f"{NOTEBOOK_PATH} (cell {index})", "exec"), workflow_module.__dict__)
    if "build_workflow" not in workflow_module.__dict__:
        raise RuntimeError("The notebook does not contain a tagged runtime cell defining build_workflow.")
    return workflow_module.__dict__


@st.cache_resource(show_spinner=False, max_entries=2)
def build_analyzer(api_key: str, model_name: str) -> Any:
    runtime = load_workflow_definitions()
    return runtime["build_workflow"](api_key=api_key, model_name=model_name)


def configured_api_key() -> str:
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        return os.getenv("OPENROUTER_API_KEY", "")


def render_banner() -> None:
    st.html(
        """
        <style>
          .meeting-banner {
            position: relative;
            overflow: hidden;
            padding: 2.25rem 2.5rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 1.25rem;
            background:
              radial-gradient(ellipse at 90% 0%, rgba(71, 214, 199, 0.26), transparent 38%),
              linear-gradient(120deg, #111a37 0%, #243d70 56%, #087f87 100%);
            box-shadow: 0 16px 42px rgba(18, 36, 73, 0.18);
            color: #f7fbff;
          }
          .meeting-banner::after {
            position: absolute;
            top: -5.5rem;
            right: 3rem;
            width: 15rem;
            height: 15rem;
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 50%;
            box-shadow: 0 0 0 2.4rem rgba(255, 255, 255, 0.035),
                        0 0 0 4.8rem rgba(255, 255, 255, 0.025);
            content: "";
            pointer-events: none;
          }
          .meeting-banner__eyebrow {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0 0 0.8rem;
            color: #aaf2e8;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-transform: uppercase;
          }
          .meeting-banner__eyebrow span {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: #57e4c7;
            box-shadow: 0 0 14px rgba(87, 228, 199, 0.8);
          }
          .meeting-banner h1 {
            position: relative;
            z-index: 1;
            width: fit-content;
            max-width: 100%;
            margin: 0;
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.1rem);
            font-weight: 750;
            letter-spacing: -0.045em;
            line-height: 1.08;
            font-variant: small-caps;
          }
          .meeting-banner__title {
            display: block;
          }
          .meeting-banner__powered {
            display: block;
            width: 100%;
            text-align: right;
            font-size: 0.55em;
            letter-spacing: 0.04em;
            white-space: nowrap;
          }
          .meeting-banner__description {
            position: relative;
            z-index: 1;
            max-width: 42rem;
            margin: 0.85rem 0 1.35rem;
            color: rgba(241, 247, 255, 0.82);
            font-size: 1.04rem;
            line-height: 1.55;
          }
          .meeting-banner__features {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
          }
          .meeting-banner__features span {
            padding: 0.4rem 0.72rem;
            border: 1px solid rgba(223, 255, 250, 0.22);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
            color: #f4fffd;
            font-size: 0.8rem;
            font-weight: 600;
          }
          @media (max-width: 600px) {
            .meeting-banner { padding: 1.6rem 1.35rem; }
          }
        </style>
        <header class="meeting-banner">
          <p class="meeting-banner__eyebrow"><span></span> Smarter meeting follow-through</p>
          <h1>
            <span class="meeting-banner__title">Meeting Transcript Insights</span>
            <span class="meeting-banner__powered">- AI Powered</span>
          </h1>
          <p class="meeting-banner__description">
            Turn conversations into clear summaries, key topics, and prioritized next steps.
          </p>
          <div class="meeting-banner__features" aria-label="Workflow capabilities">
            <span>LangGraph workflow</span>
            <span>Action owners</span>
            <span>Priority insights</span>
            <span>Local embeddings</span>
          </div>
        </header>
        """
    )


def main() -> None:
    st.set_page_config(page_title="Meeting Transcript Insights", layout="wide")
    render_banner()
    st.space("medium")

    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    api_key = configured_api_key()
    default_transcript = ""
    if TRANSCRIPT_PATH.is_file():
        default_transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8")
    transcript = st.text_area(
        "Meeting transcript",
        value=default_transcript,
        height=320,
        placeholder="Paste a transcript with or without speaker labels...",
    )

    if st.button("Analyze meeting", type="primary"):
        if not api_key.strip():
            st.error(
                "Configure OPENROUTER_API_KEY in the app's environment, .env file, or Streamlit secrets, "
                "then restart the app."
            )
            return
        if not transcript.strip():
            st.error("Enter a meeting transcript before starting the analysis.")
            return
        try:
            with st.spinner("Analyzing transcript (the first run may download the local embedding model)..."):
                analyzer = build_analyzer(api_key, model_name)
                report = analyzer.invoke({"transcript": transcript})["report"]
        except Exception as exc:
            st.error(f"Meeting analysis failed: {exc}")
            st.exception(exc)
            return

        st.subheader("Meeting Summary")
        st.write(report["meeting_summary"])

        st.subheader("Key Topics")
        for index, topic in enumerate(report["key_topics"], start=1):
            st.markdown(f"{index}. {topic}")

        st.subheader("Action Items")
        action_items = report["action_items"]
        if isinstance(action_items, str):
            st.info(action_items)
        elif action_items:
            st.dataframe(
                action_items,
                hide_index=True,
                width="stretch",
                alt="Extracted action items with owners and priority levels",
            )
        else:
            st.error("The meeting report contains an invalid action-items value.")


if __name__ == "__main__":
    main()
