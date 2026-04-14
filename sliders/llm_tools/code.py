from typing import Any

import requests

from langchain_core.tools import tool


def check_server_is_running(url: str = "http://localhost:5111/status"):
    response = requests.get(url)
    return response.json()["status"] == "running"


def execute_code(code: str, session_id: str = None, create_session: bool = False) -> dict[str, Any]:
    """
    Execute Python code through the code execution API.

    Args:
        code: The Python code to execute
        session_id: Optional session ID for stateful execution
        create_session: If True and session_id is None, creates a new session

    Returns:
        Dictionary containing execution results and session_id if applicable
    """
    payload = {"code": code}

    if session_id:
        payload["session_id"] = session_id

    if create_session:
        payload["create_session"] = True

    response = requests.post(
        "http://localhost:5111/execute",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    result = response.json()

    # Store session_id for subsequent calls
    return result


@tool
def run_python_code(code: str):
    """Run python code in a new environment with no state. The output of the code should always be printed.

    Parameters
    ----------
    code: str
        The python code to run.

    Returns
    -------
    dict:
        The output of the python code.
    """
    return execute_code(code), True
