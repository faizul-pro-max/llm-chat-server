"""libtmux wrappers for managing vllm and agent sessions."""
import libtmux


def _server() -> libtmux.Server:
    return libtmux.Server()


def create_session(name: str, command: str, log_file: str = None) -> None:
    """Create a new detached tmux session and run command inside it."""
    server = _server()
    if server.has_session(name):
        raise RuntimeError(f"Session '{name}' already exists. Run `make stop` first.")
    session = server.new_session(session_name=name, detach=True)
    pane = session.attached_pane
    if log_file:
        pane.send_keys(f"exec > >(tee -a {log_file}) 2>&1")
    pane.send_keys(command)


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
    session = server.find_where({"session_name": name})
    pane = session.attached_pane
    return pane.capture_pane(start=-lines, join=True)
