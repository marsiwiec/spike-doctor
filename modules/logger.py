from modules.models import LogEntry


class SessionLogger:
    """In-memory logger that buffers entries for display in the UI.

    This class is intentionally not thread-safe; Shiny operates on a single
    async event loop per session.
    """

    def __init__(self):
        self._entries: list[LogEntry] = []

    def log(self, level: str, abf_id: str, sweep_num, message: str) -> None:
        """Append a log entry and also print to stdout."""
        entry = LogEntry(
            level=level.upper(),
            abf_id=abf_id,
            sweep_num=int(sweep_num) if sweep_num is not None else None,
            message=message,
        )
        self._entries.append(entry)
        prefix = f"{entry.level}({entry.abf_id}"
        if entry.sweep_num is not None:
            prefix += f", Sw {entry.sweep_num}"
        prefix += ")"
        print(f"{prefix}: {entry.message}")

    def get_entries(self) -> list[LogEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# Module-level default logger for use outside of reactive contexts.
_default_logger = SessionLogger()


def get_logger() -> SessionLogger:
    return _default_logger


def set_logger(logger: SessionLogger) -> None:
    global _default_logger
    _default_logger = logger
