# ResearchPilot 🔬
### Powered by GitHub Models 

> **Agents League Hackathon — Reasoning Agents track**
> Autonomous research agent: decomposes questions → retrieves evidence → self-critiques → synthesises a structured report.

---

## Quick Start (3 steps)

### 1. Get a free GitHub token
1. Go to **github.com** → sign in
2. Click profile picture → **Settings**
3. Scroll to **Developer settings** → **Personal access tokens** → **Tokens (classic)**
4. Click **"Generate new token (classic)"** → name it `researchpilot` → click **Generate**
5. Copy the token

### 2. Set up environment
```bash
pip install -r requirements.txt
cp .env.example .env
# Open .env and paste your token as GITHUB_TOKEN=ghp_xxxx...
```

### 3. Run
```bash
python main.py "What are the latest breakthroughs in fusion energy?"
```

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────┐
│         Orchestrator            │  orchestrator.py
└──────┬──────┬──────┬────────────┘
       │      │      │
       ▼      ▼      ▼
  Planner  Researcher  Critic       agents.py
                │
         GitHub Models API
         (GPT-4o — free)
                │
                ▼
          Synthesizer
                │
                ▼
     Structured Markdown report
```

### Sub-agents

| Agent | Role |
|---|---|
| **Planner** | Decomposes query into ≤4 focused sub-questions |
| **Researcher** | Answers each sub-question with sourced evidence |
| **Critic** | Scores confidence, flags gaps, triggers re-research |
| **Synthesizer** | Writes final structured report with citations |

---

## Available free models

Change `MODEL_DEPLOYMENT_NAME` in `.env` to switch:

| Model | Good for |
|---|---|
| `gpt-4o` | Best quality (default) |
| `gpt-4o-mini` | Faster, still great |
| `meta-llama-3.1-70b-instruct` | Open source alternative |
| `mistral-large` | European option |

---

## Hackathon submission checklist

- [ ] GitHub repo is public
- [ ] README explains architecture
- [ ] Demo video recorded (5 min max, upload to YouTube/Vimeo)
- [ ] Architecture diagram included
- [ ] `.env.example` included (never commit real token)
- [ ] Submitted on Innovation Studio by June 14

---

## Project structure

```
researchpilot/
├── main.py           # Entry point + CLI + report export
├── orchestrator.py   # Pipeline: chains all agents
├── agents.py         # Planner, Researcher, Critic, Synthesizer
├── requirements.txt  # openai, rich, python-dotenv
├── .env.example      # Template (never commit .env)
└── README.md
```

---

## Built with
- [GitHub Models](https://github.com/marketplace/models) — free GPT-4o access
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- Multi-agent reasoning loop with self-critique
