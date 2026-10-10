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
NOTEBOOK_PATH = PROJECT_DIR / "AI_Meeting_Notes_Analyzer.ipynb"
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


def main() -> None:
    st.set_page_config(page_title="AI meeting notes analyzer", layout="wide")
    st.title("AI meeting notes analyzer")
    st.caption("Turn meeting transcripts into structured, actionable notes.")

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

        st.subheader("Meeting summary")
        st.write(report["meeting_summary"])

        st.subheader("Key topics")
        for index, topic in enumerate(report["key_topics"], start=1):
            st.markdown(f"{index}. {topic}")

        st.subheader("Action items")
        if report["action_items"]:
            st.dataframe(
                report["action_items"],
                hide_index=True,
                width="stretch",
                alt="Extracted action items with owners and priority levels",
            )
        else:
            st.info("No action items identified in this meeting.")

        st.subheader("Task owners")
        if report["task_owners"]:
            st.dataframe(
                report["task_owners"],
                hide_index=True,
                width="stretch",
                alt="Action items and their responsible owners",
            )
        else:
            st.write("No task owners identified.")

        st.subheader("Priority level")
        st.info(report["priority_level"])

        with st.expander("Structured report (JSON)"):
            st.json(report)


if __name__ == "__main__":
    main()
