"""
Logging utilities with optional tee-to-file.
Used by _run_v5_cowork.py for run_v5.log.
"""

import time


def make_logger(log_file: str | None = None, t0: float | None = None):
    """
    Create a log function that prints with elapsed-time prefix and optionally tees to file.

    Args:
        log_file: If provided, append each line to this file.
        t0: Start time for elapsed calculation. Defaults to time.time() at call.

    Returns:
        A function log(msg) -> str that prints and optionally writes to file.
    """
    start = t0 if t0 is not None else time.time()
    fh = open(log_file, 'w', encoding='utf-8') if log_file else None

    def log(msg: str) -> str:
        elapsed = time.time() - start
        line = f'[{elapsed:6.1f}s] {msg}'
        print(line)
        if fh is not None:
            fh.write(line + '\n')
            fh.flush()
        return line

    return log
