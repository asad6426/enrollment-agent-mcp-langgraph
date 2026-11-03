"""End-to-end tests. Each one starts the real MCP server over stdio."""
import asyncio
import json
from types import SimpleNamespace as NS

from agent_lab.graph import MAX_STEPS, build_graph
from agent_lab.planners import ClaudePlanner, ScriptedPlanner
from agent_lab.runner import claude_tools, mcp_session, run_goal, tool_caller


def run(coro):
    return asyncio.run(coro)


def with_session(fn):
    async def inner():
        async with mcp_session() as session:
            return await fn(session)

    return run(inner())


def test_server_lists_tools_over_stdio():
    async def go(session):
        return {t["name"] for t in await claude_tools(session)}

    assert with_session(go) == {
        "search_courses", "get_student_progress", "check_prerequisites", "enroll_student",
    }


def test_read_only_tool_needs_no_approval():
    def approve(_):
        raise AssertionError("approval must not be requested")

    async def go(session):
        return await run_goal("Which course teaches pytorch?", ScriptedPlanner(), session, approve)

    result = with_session(go)
    assert "DL301" in result["final"]


def test_approved_enrolment_changes_the_record():
    requests = []

    def approve(request):
        requests.append(request)
        return True

    async def go(session):
        result = await run_goal("Enroll S100 in DL301", ScriptedPlanner(), session, approve)
        _, progress = await tool_caller(session)("get_student_progress", {"student_id": "S100"})
        return result, json.loads(progress)

    result, progress = with_session(go)
    assert requests == [{"tool": "enroll_student", "args": {"student_id": "S100", "course_id": "DL301"}}]
    assert progress["enrolled"] == ["DL301"]
    assert "now enrolled" in result["final"]


def test_denied_enrolment_leaves_the_record_unchanged():
    async def go(session):
        result = await run_goal("Enroll S100 in DL301", ScriptedPlanner(), session, lambda _: False)
        _, progress = await tool_caller(session)("get_student_progress", {"student_id": "S100"})
        return result, json.loads(progress)

    result, progress = with_session(go)
    assert progress["enrolled"] == []
    assert "not enrolled" in result["final"]


def test_reviewer_blocks_missing_prerequisites_before_asking_a_human():
    def approve(_):
        raise AssertionError("approval must not be requested")

    async def go(session):
        result = await run_goal("Enroll S200 in DL301", ScriptedPlanner(), session, approve)
        _, progress = await tool_caller(session)("get_student_progress", {"student_id": "S200"})
        return result, json.loads(progress)

    result, progress = with_session(go)
    assert progress["enrolled"] == []
    assert "Reviewer blocked" in result["final"]
    assert "ML201" in result["final"]


def test_unknown_student_is_reported_as_failure():
    async def go(session):
        return await run_goal("Enroll S999 in DL301", ScriptedPlanner(), session, lambda _: True)

    assert "not enrolled" in with_session(go)["final"]


def test_server_enforces_prerequisites_if_the_graph_is_bypassed():
    async def go(session):
        return await tool_caller(session)("enroll_student", {"student_id": "S200", "course_id": "DL301"})

    ok, text = with_session(go)
    assert not ok and "Missing prerequisites" in text


def test_step_limit_stops_a_looping_planner():
    class Looping:
        def next_action(self, goal, history):
            return {"type": "tool", "name": "search_courses", "args": {"topic": "x"}}

    async def fake_tool(name, args):
        return True, "[]"

    async def go():
        graph = build_graph(Looping(), fake_tool)
        return await graph.ainvoke({"goal": "x", "history": [], "audit": []},
                                   {"configurable": {"thread_id": "t"}, "recursion_limit": 100})

    result = run(go())
    assert result["final"] == f"Stopped after {MAX_STEPS} steps."


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_claude_planner_round_trips_tool_results_with_a_fake_client():
    tool_use = NS(type="tool_use", id="toolu_1", name="search_courses", input={"topic": "rag"})
    client = FakeClient([
        NS(stop_reason="tool_use", content=[NS(type="thinking"), tool_use]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="Take AG401.")]),
    ])
    planner = ClaudePlanner(tools=[], client=client)
    action = planner.next_action("Which course covers RAG?", [])
    assert action == {"type": "tool", "name": "search_courses", "args": {"topic": "rag"}}

    history = [{"tool": "search_courses", "args": {"topic": "rag"}, "ok": True, "result": "[]"}]
    assert planner.next_action("Which course covers RAG?", history) == {"type": "final", "text": "Take AG401."}

    sent = client.calls[1]["messages"]
    assert sent[1]["content"][0].type == "thinking"
    assert sent[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert client.calls[0]["model"] == "claude-opus-5-5"
    assert client.calls[0]["tool_choice"]["type"] == "auto"


def test_claude_planner_handles_refusal():
    client = FakeClient([NS(stop_reason="refusal", content=[])])
    action = ClaudePlanner(tools=[], client=client).next_action("x", [])
    assert action["type"] == "final"
