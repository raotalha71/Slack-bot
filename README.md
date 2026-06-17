# Slack AI Proposal Generator

A multi-agent AI system that automatically generates professional business proposals from sales call transcripts. Built with LangGraph, FastAPI, Qdrant, and Slack Bolt.

## Features

- **Multi-Agent Pipeline:** 4 specialized agents (Intake, Research, Writer, Revision) working together.
- **Human-in-the-Loop:** Pauses at key decision points for user confirmation (intake extraction, web search approval, tone selection).
- **Self-Learning RAG:** Automatically saves novel generated proposals back to Qdrant to improve future context.
- **Smart Save Gate:** Deduplicates proposals before saving to keep the vector database clean (skips if similarity > 0.85).
- **Web Search Fallback:** Uses Tavily API to search the internet if the internal knowledge base lacks context.
- **Cross-Session Memory:** Remembers transcripts and drafts via an SQLite session store, even if the user returns days later.
- **Surgical Revisions:** Asks follow-up questions and specifically edits single sections of the document without rewriting the entire proposal.
- **DOCX Generation:** Converts the final Markdown draft into a styled, professional Word document.

---

## 🏗 Architecture Overview

The system uses a stateful LangGraph pipeline (`ProposalState`) across two distinct graphs:

1. **Generation Graph:**
   `Slack File Upload` → `Intake Agent` → *(Interrupt: Confirm)* → `Research Agent` → *(Interrupt: Web Search?)* → *(Interrupt: Pick Tone)* → `Writer Agent` → `Smart Save Gate` → `Generate DOCX`
2. **Revision Graph:**
   `Slack Thread Reply` → `Revision Agent` (Answers questions via Transcript RAG or edits specific sections) → `Regenerate DOCX`

*For an in-depth breakdown of technical decisions, see [docs/decision_doc.md](docs/decision_doc.md).*

---

## 🚀 Quickstart Guide

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for Qdrant)
- A Slack Workspace with an installed App (requires Bot Token and Signing Secret)
- API Keys: Groq (LLM) and Tavily (Web Search)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/raotalha71/Slack-bot.git
cd Slack-bot

# Create and activate virtual environment
python3.12 -m venv --copies venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```
Edit `.env` and add your keys:
- `GROQ_API_KEY`: Get from [Groq Console](https://console.groq.com/)
- `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com/)
- `SLACK_BOT_TOKEN`: Starts with `xoxb-...`
- `SLACK_SIGNING_SECRET`: Get from Slack App Basic Information

### 3. Start Infrastructure (Qdrant & SQLite)

Start Qdrant using Docker Compose:
```bash
docker-compose up -d qdrant
```

*(Note: The main application runs locally via Uvicorn during development. You can also run the entire stack via `docker-compose up`.)*

### 4. Run the Application

```bash
uvicorn main:api --reload --port 8000
```

On startup, the app will:
1. Initialize the SQLite database.
2. Connect to Qdrant.
3. Automatically download the embedding model (`all-MiniLM-L6-v2`).
4. Ingest the seed proposals from `seed_data/` into Qdrant.

### 5. Connect Slack (ngrok)

Since Slack needs a public URL to send events, use ngrok to expose your local server:
```bash
ngrok http 8000
```
- Copy the `https://...` Forwarding URL.
- Go to your Slack App configuration -> **Event Subscriptions**.
- Enable Events and paste the URL: `https://<your-ngrok-url>/slack/events`.
- Subscribe to bot events: `file_shared`, `message.channels`, `message.im`.

---

## 🛠 How to Use the Bot

1. **Start a Generation:** Upload a `.txt` or `.md` transcript file directly to the Slack channel where the bot is invited.
2. **Confirm Extraction:** The bot will reply in a thread with extracted info. Reply `ok` or specify corrections.
3. **Handle Fallbacks:** If the bot doesn't find relevant past proposals, it will ask if you want to search the web. Reply `yes` or `no`.
4. **Select Tone:** Reply with a number (1-4) to choose the proposal tone.
5. **Review DOCX:** The bot will generate and upload the final `.docx` proposal.
6. **Revise:** Reply in the thread to ask questions (`What was their budget?`) or request edits (`Change the timeline section to be 6 months`).

---

## 📁 Repository Structure

```text
Slack-bot/
├── app/
│   ├── agents/          # LangGraph agents (Intake, Research, Writer, Revision)
│   ├── models/          # Pydantic schemas and TypedDict state
│   ├── rag/             # Qdrant integration, chunking, and ingestion
│   ├── services/        # Orchestration layer (ProposalService)
│   ├── slack/           # Slack event handlers and bolt app
│   └── tools/           # Web search (Tavily) and DOCX generation
├── data/                # SQLite database (auto-created)
├── docs/                # Architecture decision document
├── scripts/             # Standalone scripts (seed ingestion)
├── seed_data/           # Assessment-provided sample proposals
├── .env.example         # Environment variable template
├── docker-compose.yml   # Infrastructure setup
├── main.py              # FastAPI entry point
└── requirements.txt     # Python dependencies
```