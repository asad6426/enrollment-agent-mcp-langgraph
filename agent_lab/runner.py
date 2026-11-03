"""Connects to the MCP server over stdio and drives the graph to completion."""
import sys
from contextlib import asynccontextmanager
from typing import Callable

from langgraph.types import Command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .graph import build_graph
from .planners import ClaudePlanner, Planner, ScriptedPlanner


@asynccontextmanager
async def mcp_session():
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent_lab.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def tool_caller(session: ClientSession):
    async def call_tool(name: str, args: dict) -> tuple[bool, str]:
        result = await session.call_tool(name, args)
        text = "".join(block.text for block in result.content if block.type == "text")
        return not result.isError, text

    return call_tool


async def run_goal(goal: str, planner: Planner, session: ClientSession,
                   approve: Callable[[dict], bool], thread_id: str = "demo") -> dict:
    graph = build_graph(planner, tool_caller(session))
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke({"goal": goal, "history": [], "audit": []}, config)
    while True:
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return result
        request = snapshot.tasks[0].interrupts[0].value
        result = await graph.ainvoke(Command(resume=approve(request)), config)


async def claude_tools(session: ClientSession) -> list[dict]:
    listed = await session.list_tools()
    return [{"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            for t in listed.tools]


async def make_planner(use_claude: bool, session: ClientSession) -> Planner:
    return ClaudePlanner(await claude_tools(session)) if use_claude else ScriptedPlanner()
