"""
orchestrator.py — ResearchPilot pipeline (GitHub Models edition)
Chains: Planner → Researcher(s) → Critic → (re-research loop) → Synthesizer
"""
from __future__ import annotations

import os

from rich.console import Console
from rich.panel import Panel

from agents import (
    CriticReport, FinalReport, ResearchPlan, ResearchResult,
    make_client, run_critic, run_planner, run_researcher, run_synthesizer,
)

console = Console()
ENABLE_CRITIC_LOOP = os.getenv("ENABLE_CRITIC_LOOP", "true").lower() == "true"
MAX_LOOPS = 2


def run_pipeline(query: str) -> FinalReport:
    console.print(Panel(
        f"[bold]Query:[/bold] {query}",
        title="[purple]ResearchPilot — GitHub Models[/purple]",
        border_style="purple",
    ))

    client = make_client()
    console.print("[dim]Connected to GitHub Models (free tier) ✓[/dim]\n")

    # 1. Plan
    plan: ResearchPlan = run_planner(client, query)
    console.print()
    for i, q in enumerate(plan.sub_questions, 1):
        console.print(f"  [dim]{i}.[/dim] {q}")
    console.print()

    # 2. Research each sub-question
    results: list[ResearchResult] = []
    for sub_q in plan.sub_questions:
        results.append(run_researcher(client, sub_q))
    console.print()

    # 3. Critic evaluation
    critic: CriticReport = run_critic(client, plan, results)

    # 4. Optional re-research loop
    if ENABLE_CRITIC_LOOP:
        loops = 0
        while critic.needs_more_research and loops < MAX_LOOPS:
            loops += 1
            console.print(f"\n[yellow]↩  Re-research loop {loops}/{MAX_LOOPS} — gaps found:[/yellow]")
            for gap in critic.gaps:
                console.print(f"   • {gap}")
            for gap in critic.gaps:
                results.append(run_researcher(client, gap))
            critic = run_critic(client, plan, results)

    # 5. Synthesize
    console.print()
    return run_synthesizer(client, plan, results, critic)
