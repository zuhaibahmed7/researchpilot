"""
agents.py — ResearchPilot sub-agents (GitHub Models edition)
Uses the OpenAI SDK pointed at GitHub's free inference endpoint.
No Azure subscription required — just a free GitHub token.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI
from rich.console import Console

console = Console()

# ── GitHub Models client ─────────────────────────────────────────────────────

def make_client() -> OpenAI:
    """Return an OpenAI client pointed at GitHub Models (free)."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN not set. Get one at: "
            "github.com → Settings → Developer settings → Personal access tokens"
        )
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

MODEL = lambda: os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o")
MAX_SUB_QUESTIONS = int(os.environ.get("MAX_SUB_QUESTIONS", "4"))


# ── Shared helper: one-shot LLM call ─────────────────────────────────────────

def _call(client: OpenAI, system: str, user: str, max_tokens: int = 2000) -> str:
    """Simple single-turn call. Returns the assistant's text response."""
    response = client.chat.completions.create(
        model=MODEL(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ResearchPlan:
    original_query: str
    sub_questions: list[str]
    scope_notes: str = ""


@dataclass
class ResearchResult:
    sub_question: str
    answer: str
    sources: list[str] = field(default_factory=list)


@dataclass
class CriticReport:
    gaps: list[str]
    contradictions: list[str]
    confidence: float
    needs_more_research: bool


@dataclass
class FinalReport:
    title: str
    summary: str
    sections: list[dict[str, str]]
    sources: list[str]
    confidence: float


# ── 1. Planner ────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = f"""You are a research planning expert.
Decompose the user's query into at most {MAX_SUB_QUESTIONS} precise, non-overlapping sub-questions.
Respond ONLY with valid JSON — no markdown, no preamble:
{{
  "sub_questions": ["question 1", "question 2", ...],
  "scope_notes": "one sentence on what is in/out of scope"
}}"""


def run_planner(client: OpenAI, query: str) -> ResearchPlan:
    console.print("[bold cyan]🗺  Planner[/bold cyan] → decomposing query…")
    raw = _call(client, PLANNER_SYSTEM, f"Research query: {query}")
    # Strip markdown fences if model adds them
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat whole query as one question
        data = {"sub_questions": [query], "scope_notes": ""}

    plan = ResearchPlan(
        original_query=query,
        sub_questions=data.get("sub_questions", [query]),
        scope_notes=data.get("scope_notes", ""),
    )
    console.print(f"  [green]✓[/green] {len(plan.sub_questions)} sub-questions")
    return plan


# ── 2. Researcher ─────────────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a meticulous research assistant with deep knowledge across many domains.
Answer the question thoroughly using your training knowledge.
Respond ONLY with valid JSON — no markdown, no preamble:
{
  "answer": "detailed factual answer (3-5 paragraphs)",
  "sources": ["source description 1", "source description 2", ...]
}
Sources should be specific (e.g. 'Nature journal studies on X', 'MIT research on Y', 'Industry reports from Z')."""


def run_researcher(client: OpenAI, sub_question: str) -> ResearchResult:
    short = sub_question[:65] + "…" if len(sub_question) > 65 else sub_question
    console.print(f"  [bold cyan]🔍 Researcher[/bold cyan] → {short}")
    raw = _call(client, RESEARCHER_SYSTEM, sub_question, max_tokens=1500)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        answer = data.get("answer", raw)
        sources = data.get("sources", [])
    except json.JSONDecodeError:
        answer = raw
        sources = []

    console.print(f"    [green]✓[/green] answered ({len(sources)} sources)")
    return ResearchResult(sub_question=sub_question, answer=answer, sources=sources)


# ── 3. Critic ─────────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a rigorous research critic.
Given a research plan and draft answers, evaluate:
1. Gaps — aspects of the original query still unanswered
2. Contradictions — inconsistencies between answers
3. Overall confidence (0.0 = unreliable, 1.0 = comprehensive)
4. Whether more research is needed

Respond ONLY with valid JSON — no markdown, no preamble:
{
  "gaps": ["gap 1", ...],
  "contradictions": ["contradiction 1", ...],
  "confidence": 0.75,
  "needs_more_research": false
}"""


def run_critic(client: OpenAI, plan: ResearchPlan, results: list[ResearchResult]) -> CriticReport:
    console.print("[bold cyan]⚖  Critic[/bold cyan] → evaluating coverage…")
    payload = {
        "original_query": plan.original_query,
        "sub_questions": plan.sub_questions,
        "answers": [{"question": r.sub_question, "answer": r.answer[:600]} for r in results],
    }
    raw = _call(client, CRITIC_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=800)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"gaps": [], "contradictions": [], "confidence": 0.7, "needs_more_research": False}

    report = CriticReport(
        gaps=data.get("gaps", []),
        contradictions=data.get("contradictions", []),
        confidence=float(data.get("confidence", 0.7)),
        needs_more_research=bool(data.get("needs_more_research", False)),
    )
    console.print(
        f"  [green]✓[/green] confidence={report.confidence:.0%}  "
        f"gaps={len(report.gaps)}  more_needed={report.needs_more_research}"
    )
    return report


# ── 4. Synthesizer ────────────────────────────────────────────────────────────

SYNTHESIZER_SYSTEM = """You are a professional research writer.
Write a clear, structured research report from the findings provided.
Respond ONLY with valid JSON — no markdown, no preamble:
{
  "title": "Report title",
  "summary": "2-3 sentence executive summary",
  "sections": [
    {"heading": "Section heading", "body": "2-4 paragraphs of well-written prose"}
  ]
}
Rules: professional English, acknowledge known gaps, no filler phrases."""


def run_synthesizer(
    client: OpenAI,
    plan: ResearchPlan,
    results: list[ResearchResult],
    critic: CriticReport,
) -> FinalReport:
    console.print("[bold cyan]✍  Synthesizer[/bold cyan] → writing report…")
    all_sources = sorted({s for r in results for s in r.sources})
    payload = {
        "original_query": plan.original_query,
        "findings": [{"question": r.sub_question, "answer": r.answer[:400]} for r in results],
        "critic": {"gaps": critic.gaps, "confidence": critic.confidence},
    }
    raw = _call(client, SYNTHESIZER_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=1500)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "title": plan.original_query,
            "summary": "Research complete.",
            "sections": [{"heading": "Findings", "body": raw}],
        }

    report = FinalReport(
        title=data.get("title", plan.original_query),
        summary=data.get("summary", ""),
        sections=data.get("sections", []),
        sources=all_sources,
        confidence=critic.confidence,
    )
    console.print(f"  [green]✓[/green] {len(report.sections)} sections written")
    return report
