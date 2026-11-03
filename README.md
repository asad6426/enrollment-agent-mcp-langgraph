# Enrollment Agent: MCP and LangGraph

[![CI](https://github.com/asad6426/enrollment-agent-mcp-langgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/asad6426/enrollment-agent-mcp-langgraph/actions/workflows/ci.yml)

An agent that looks up courses and enrolls students, built to teach the parts of an agent that matter once it can change real records: tool boundaries, a policy check, human approval and an audit trail. The students and courses are made up.

## How it works

```text
          +---------------------------------------+
          v                                       |
START -> advisor -> reviewer -> tools (via MCP) --+
           |            |  ^          ^
           v            v  |          |
          END        blocked|     approval (human)
                  (back to advisor)
```

- **Advisor** chooses the next action. It is either rule-based or Claude, behind one interface ([`agent_lab/planners.py`](agent_lab/planners.py)).
- **Reviewer** checks prerequisites before any high-impact tool, and blocks ineligible requests before a human is asked ([`agent_lab/graph.py`](agent_lab/graph.py)).
- **Approval** pauses the graph with LangGraph's `interrupt` until a person decides.
- **Tools** run on a real MCP server, as a separate process over stdio ([`agent_lab/mcp_server.py`](agent_lab/mcp_server.py)).
- **Safety limits:** the server enforces prerequisites again in case the graph is bypassed, and the graph stops after 8 steps.

## What a run looks like

Output from the rule-based advisor, as produced in CI:

```text
approval requested: enroll_student {'student_id': 'S100', 'course_id': 'DL301'} -> approve
goal: Enroll S100 in DL301
audit: get_student_progress {"student_id": "S100"} ok
audit: approved enroll_student
audit: enroll_student {"student_id": "S100", "course_id": "DL301"} ok
answer: S100 is now enrolled in DL301.

approval requested: enroll_student {'student_id': 'S100', 'course_id': 'DL301'} -> deny
goal: Enroll S100 in DL301
audit: get_student_progress {"student_id": "S100"} ok
audit: enroll_student {"student_id": "S100", "course_id": "DL301"} failed
answer: S100 was not enrolled in DL301: A human denied this action.

goal: Enroll S200 in DL301
audit: get_student_progress {"student_id": "S200"} ok
audit: enroll_student {"student_id": "S200", "course_id": "DL301"} failed
answer: S200 was not enrolled in DL301: Reviewer blocked: {"eligible": false, "missing": ["ML201"]}
```

## Tests

10 tests in [`tests/test_agent.py`](tests/test_agent.py). The workflow tests start the real MCP server over stdio. They cover approved, denied and blocked enrollments, an unknown student, the server's own prerequisite check and a planner stuck in a loop.

## Limits

- The Claude planner is tested against a fake client. Its tool-result round trip and refusal handling are covered, but this repository's CI does not call the live API.
- The reviewer is a deterministic policy step, not a second model.
- Data lives in the server's memory and resets when the process ends.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q
python -m agent_lab "Enroll S100 in DL301"
python -m agent_lab "Enroll S100 in DL301" --deny
python -m agent_lab "Which course teaches pytorch?"
```

To plan with Claude, set `ANTHROPIC_API_KEY` and add `--claude`. The planner defaults to `claude-opus-5-5` and enables server-side refusal fallbacks.

CI last ran with `langgraph` 1.2.19, `mcp` 2.3.0 and `anthropic` 2.4.0. Built with Claude Code.
