"""libtmux wrappers for managing vllm and agent sessions."""
import os
import shlex

import libtmux


def _server() -> libtmux.Server:
    return libtmux.Server()


def create_session(name: str, command: str, log_file: str = None) -> None:
    """Create a new detached tmux session and run command inside it.

    The pane runs an interactive login shell, so on some hosts (e.g. Vast.ai)
    ~/.bashrc resets the working directory after libtmux's start_directory takes
    effect. To be robust we send a single atomic command that first cd's back to
    the project root, then sets up logging, then runs the command — so nothing
    depends on the shell's startup behaviour.
    """
    server = _server()
    if server.has_session(name):
        raise RuntimeError(f"Session '{name}' already exists. Run `make stop` first.")

    cwd = os.getcwd()
    session = server.new_session(session_name=name, detach=True, start_directory=cwd)
    pane = session.active_pane

    prefix = f"cd {shlex.quote(cwd)}"
    if log_file:
        log_abs = os.path.abspath(log_file)
        prefix += f" && exec > >(tee -a {shlex.quote(log_abs)}) 2>&1"
    pane.send_keys(f"{prefix} ; {command}")


def is_running(name: str) -> bool:
    """Return True if a tmux session with this name exists."""
    return _server().has_session(name)


def kill_session(name: str) -> None:
    """Kill the named session. No-op if it does not exist."""
    server = _server()
    if server.has_session(name):
        server.kill_session(name)


def capture_pane(name: str, lines: int = 50) -> str:
    """Return the last N lines of output from the named session."""
    server = _server()
    if not server.has_session(name):
        return ""
    session = server.sessions.get(session_name=name)
    if session is None:
        return ""
    pane = session.active_pane
    output = pane.capture_pane(start=-lines)
    if isinstance(output, list):
        return "\n".join(output)
    return output or ""
