"""
server.py — ResearchPilot FastAPI backend
Streams agent progress to the frontend in real time using Server-Sent Events.

Run with:
    pip install fastapi uvicorn
    python server.py
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="ResearchPilot API")

# Allow the frontend (index.html) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Streaming research endpoint ───────────────────────────────────────────────

@app.post("/research")
async def research(req: QueryRequest):
    """
    Run the full research pipeline and stream progress events to the frontend.
    Events: log, plan, subquestion_done, critic, reresearch, report, done, error
    """
    async def generate():
        try:
            from openai import OpenAI
            import agents as ag

            query = req.query.strip()
            if not query:
                yield sse("error", {"message": "Empty query"})
                return

            client = ag.make_client()
            yield sse("log", {"tag": "SYSTEM", "msg": "Connected to GitHub Models ✓"})
            await asyncio.sleep(0.05)

            # ── 1. Planner ────────────────────────────────────────────────────
            yield sse("log", {"tag": "PLANNER", "msg": "Decomposing query into sub-questions…"})
            await asyncio.sleep(0.05)

            plan = await asyncio.get_event_loop().run_in_executor(
                None, ag.run_planner, client, query
            )
            yield sse("plan", {
                "sub_questions": plan.sub_questions,
                "scope_notes": plan.scope_notes,
            })
            yield sse("log", {
                "tag": "PLANNER",
                "msg": f"Generated {len(plan.sub_questions)} sub-questions"
            })
            await asyncio.sleep(0.05)

            # ── 2. Researcher ─────────────────────────────────────────────────
            results = []
            for i, sub_q in enumerate(plan.sub_questions):
                yield sse("log", {
                    "tag": "RESEARCHER",
                    "msg": f"Q{i+1}: {sub_q[:70]}…"
                })
                await asyncio.sleep(0.05)

                result = await asyncio.get_event_loop().run_in_executor(
                    None, ag.run_researcher, client, sub_q
                )
                results.append(result)

                yield sse("subquestion_done", {
                    "index": i,
                    "sources_count": len(result.sources)
                })
                yield sse("log", {
                    "tag": "RESEARCHER",
                    "msg": f"Q{i+1} answered — {len(result.sources)} sources found"
                })
                await asyncio.sleep(0.05)

            # ── 3. Critic ─────────────────────────────────────────────────────
            yield sse("log", {"tag": "CRITIC", "msg": "Evaluating coverage and consistency…"})
            await asyncio.sleep(0.05)

            critic = await asyncio.get_event_loop().run_in_executor(
                None, ag.run_critic, client, plan, results
            )
            yield sse("critic", {
                "confidence": critic.confidence,
                "gaps": critic.gaps,
                "contradictions": critic.contradictions,
                "needs_more_research": critic.needs_more_research,
            })
            yield sse("log", {
                "tag": "CRITIC",
                "msg": f"Confidence: {critic.confidence:.0%} — gaps: {len(critic.gaps)}"
            })
            await asyncio.sleep(0.05)

            # ── 4. Re-research loop ───────────────────────────────────────────
            enable_loop = os.getenv("ENABLE_CRITIC_LOOP", "true").lower() == "true"
            loop_count = 0
            max_loops = 2

            if enable_loop:
                while critic.needs_more_research and loop_count < max_loops:
                    loop_count += 1
                    yield sse("reresearch", {"loop": loop_count, "gaps": critic.gaps})
                    yield sse("log", {
                        "tag": "CRITIC",
                        "msg": f"Re-research loop {loop_count}/{max_loops} — filling gaps…"
                    })
                    await asyncio.sleep(0.05)

                    for gap in critic.gaps:
                        gap_result = await asyncio.get_event_loop().run_in_executor(
                            None, ag.run_researcher, client, gap
                        )
                        results.append(gap_result)
                        yield sse("log", {
                            "tag": "RESEARCHER",
                            "msg": f"Gap answered — {len(gap_result.sources)} sources"
                        })
                        await asyncio.sleep(0.05)

                    critic = await asyncio.get_event_loop().run_in_executor(
                        None, ag.run_critic, client, plan, results
                    )
                    yield sse("log", {
                        "tag": "CRITIC",
                        "msg": f"Re-eval confidence: {critic.confidence:.0%}"
                    })
                    await asyncio.sleep(0.05)

            # ── 5. Synthesizer ────────────────────────────────────────────────
            yield sse("log", {"tag": "SYNTHESIZER", "msg": "Writing final report…"})
            await asyncio.sleep(0.05)

            report = await asyncio.get_event_loop().run_in_executor(
                None, ag.run_synthesizer, client, plan, results, critic
            )

            # Save report to disk
            slug = query[:40].lower().replace(" ", "_").replace("/", "-").strip("_")
            filename = f"report_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            yield sse("report", {
                "title": report.title,
                "summary": report.summary,
                "sections": report.sections,
                "sources": report.sources,
                "confidence": report.confidence,
                "filename": filename,
            })
            yield sse("log", {
                "tag": "SYNTHESIZER",
                "msg": f"Report complete — {len(report.sections)} sections"
            })
            yield sse("done", {"filename": filename})

        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🔬 ResearchPilot server starting...")
    print("   Open http://localhost:8000 in your browser\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
