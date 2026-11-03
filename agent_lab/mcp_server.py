"""MCP server over stdio exposing a small, made-up course catalogue."""
import copy
import json

from mcp.server.fastmcp import FastMCP

COURSES = {
    "PY101": {"title": "Python Foundations", "topics": ["python"], "prerequisites": []},
    "ML201": {"title": "Machine Learning with scikit-learn", "topics": ["machine learning", "scikit-learn"], "prerequisites": ["PY101"]},
    "DL301": {"title": "Deep Learning with PyTorch", "topics": ["deep learning", "pytorch", "cnn", "transformers"], "prerequisites": ["ML201"]},
    "AG401": {"title": "LLM Apps and Agents", "topics": ["llm", "rag", "mcp", "langgraph", "agents"], "prerequisites": ["PY101"]},
}
INITIAL_STUDENTS = {
    "S100": {"name": "Sample Learner A", "completed": ["PY101", "ML201"], "enrolled": []},
    "S200": {"name": "Sample Learner B", "completed": [], "enrolled": []},
}
STUDENTS = copy.deepcopy(INITIAL_STUDENTS)

mcp = FastMCP("course-catalogue")


def missing_prerequisites(student_id: str, course_id: str) -> list[str]:
    completed = STUDENTS[student_id]["completed"]
    return [p for p in COURSES[course_id]["prerequisites"] if p not in completed]


def _require(student_id: str, course_id: str) -> None:
    if student_id not in STUDENTS:
        raise ValueError(f"Unknown student: {student_id}")
    if course_id not in COURSES:
        raise ValueError(f"Unknown course: {course_id}")


@mcp.tool()
def search_courses(topic: str) -> str:
    """Find courses whose title or topics mention the given topic."""
    needle = topic.lower()
    found = [
        {"id": cid, "title": c["title"], "prerequisites": c["prerequisites"]}
        for cid, c in COURSES.items()
        if needle in c["title"].lower() or any(needle in t for t in c["topics"])
    ]
    return json.dumps(found)


@mcp.tool()
def get_student_progress(student_id: str) -> str:
    """Return the courses a student has completed and is enrolled in."""
    if student_id not in STUDENTS:
        raise ValueError(f"Unknown student: {student_id}")
    return json.dumps(STUDENTS[student_id])


@mcp.tool()
def check_prerequisites(student_id: str, course_id: str) -> str:
    """Report which prerequisites a student still needs for a course."""
    _require(student_id, course_id)
    missing = missing_prerequisites(student_id, course_id)
    return json.dumps({"eligible": not missing, "missing": missing})


@mcp.tool()
def enroll_student(student_id: str, course_id: str) -> str:
    """Enrol a student in a course. This changes records and needs human approval."""
    _require(student_id, course_id)
    missing = missing_prerequisites(student_id, course_id)
    if missing:
        raise ValueError(f"Missing prerequisites: {', '.join(missing)}")
    if course_id not in STUDENTS[student_id]["enrolled"]:
        STUDENTS[student_id]["enrolled"].append(course_id)
    return json.dumps({"enrolled": STUDENTS[student_id]["enrolled"]})


if __name__ == "__main__":
    mcp.run(transport="stdio")
