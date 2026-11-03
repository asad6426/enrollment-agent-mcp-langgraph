"""Advisor, reviewer and approval nodes wired together with LangGraph."""
import json
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .planners import Planner

HIGH_IMPACT_TOOLS = {"enroll_student"}
MAX_STEPS = 8

ToolCaller = Callable[[str, dict], Awaitable[tuple[bool, str]]]


class AgentState(TypedDict, total=False):
    goal: str
    history: list[dict]
    pending: dict | None
    approved: bool
    steps: int
    final: str
    audit: list[str]


def build_graph(planner: Planner, call_tool: ToolCaller):
    async def advisor(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= MAX_STEPS:
            return {"final": f"Stopped after {MAX_STEPS} steps.", "pending": None}
        action = planner.next_action(state["goal"], state.get("history", []))
        if action["type"] == "final":
            return {"final": action["text"], "pending": None}
        return {"pending": action, "steps": steps + 1, "approved": False}

    async def reviewer(state: AgentState) -> dict:
        """A second agent role: checks policy before a high-impact tool can run."""
        action = state["pending"]
        if action["name"] not in HIGH_IMPACT_TOOLS:
            return {"approved": True}
        ok, result = await call_tool("check_prerequisites", action["args"])
        if ok and json.loads(result)["eligible"]:
            return {}
        reason = f"Reviewer blocked: {result}"
        return _record(state, action, False, reason)

    async def approval(state: AgentState) -> dict:
        action = state["pending"]
        decision: Any = interrupt({"tool": action["name"], "args": action["args"]})
        if decision is True:
            return {"approved": True, "audit": _audit(state, f"approved {action['name']}")}
        return _record(state, action, False, "A human denied this action.")

    async def tools(state: AgentState) -> dict:
        action = state["pending"]
        ok, result = await call_tool(action["name"], action["args"])
        return _record(state, action, ok, result)

    def after_advisor(state: AgentState) -> str:
        return "reviewer" if state.get("pending") else END

    def after_reviewer(state: AgentState) -> str:
        if state.get("pending") is None:
            return "advisor"
        return "tools" if state.get("approved") else "approval"

    def after_approval(state: AgentState) -> str:
        return "tools" if state.get("approved") and state.get("pending") else "advisor"

    graph = StateGraph(AgentState)
    graph.add_node("advisor", advisor)
    graph.add_node("reviewer", reviewer)
    graph.add_node("approval", approval)
    graph.add_node("tools", tools)
    graph.add_edge(START, "advisor")
    graph.add_conditional_edges("advisor", after_advisor, ["reviewer", END])
    graph.add_conditional_edges("reviewer", after_reviewer, ["advisor", "tools", "approval"])
    graph.add_conditional_edges("approval", after_approval, ["advisor", "tools"])
    graph.add_edge("tools", "advisor")
    return graph.compile(checkpointer=MemorySaver())


def _audit(state: AgentState, line: str) -> list[str]:
    return [*state.get("audit", []), line]


def _record(state: AgentState, action: dict, ok: bool, result: str) -> dict:
    event = {"tool": action["name"], "args": action["args"], "ok": ok, "result": result}
    status = "ok" if ok else "failed"
    return {
        "history": [*state.get("history", []), event],
        "pending": None,
        "approved": False,
        "audit": _audit(state, f"{action['name']} {json.dumps(action['args'])} {status}"),
    }
