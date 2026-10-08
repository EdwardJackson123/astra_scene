"""Bounded server stages; terminate the owned process group on timeout."""
import os, signal, subprocess


def run(command, *, timeout=7200, check=True, **kwargs):
    if timeout <= 0 or timeout > 86400: raise ValueError('invalid stage timeout')
    if os.name != 'posix':
        return subprocess.run(command, timeout=timeout, check=check, **kwargs)
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            # Descendants can survive even when their parent has exited.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            raise
        if check and code: raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
