# AI meeting notes analyzer

This project analyzes a meeting transcript with a LangGraph workflow. Topic extraction, a 3–5 sentence summary, action extraction, and priority classification are separate structured-output agents. If no actions are found, the graph skips priority classification. OpenRouter supplies the chat model; transcript embeddings are generated locally with `sentence-transformers/all-MiniLM-L6-v2`.

## Setup

Python 3.14 is required. In PowerShell from this folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `OPENROUTER_API_KEY` in the environment, add it to a local `.env` file, or configure it in `.streamlit/secrets.toml`. The API key is read from configuration and is not displayed or editable in the app:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

The app has no sidebar and defaults to `nvidia/nemotron-3-ultra-550b-a55b:free`. To use another model, set `OPENROUTER_MODEL` in the app environment.

## Run the notebook

Open [AI_Meeting_Notes_Analyzer.ipynb](./AI_Meeting_Notes_Analyzer.ipynb) with the project virtual environment as its kernel. Run the workflow-definition cell, then the analysis cell. The first run downloads the local embedding model.

## Run the Streamlit app

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The app uses the meeting transcript in `Meeting_Transcript.txt` as its initial text; paste another transcript in the editor to analyze it instead. The workflow code lives in the notebook's `runtime`-tagged cell, which the Streamlit app loads directly.
