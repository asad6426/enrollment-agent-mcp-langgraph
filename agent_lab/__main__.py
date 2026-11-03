"""Usage: python -m agent_lab "Enroll S100 in DL301" [--deny] [--claude]"""
import asyncio
import sys

from .runner import make_planner, mcp_session, run_goal


async def main() -> None:
    goals = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(goals) != 1:
        sys.exit(__doc__)

    def approve(request: dict) -> bool:
        decision = "--deny" not in sys.argv
        print(f"approval requested: {request['tool']} {request['args']} -> {'approve' if decision else 'deny'}")
        return decision

    async with mcp_session() as session:
        planner = await make_planner("--claude" in sys.argv, session)
        result = await run_goal(goals[0], planner, session, approve)
    print(f"goal: {goals[0]}")
    for line in result["audit"]:
        print(f"audit: {line}")
    print(f"answer: {result['final']}")


if __name__ == "__main__":
    asyncio.run(main())
