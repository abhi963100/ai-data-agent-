"""
ollama_safe.py
-------------------------------------------------
Global safe wrapper for Ollama subprocess calls
Fixes Windows UnicodeDecodeError (cp1252)
NO changes required in existing application logic
-------------------------------------------------
"""

import subprocess as _subprocess

# ---------------------------
# SAVE ORIGINAL subprocess.run
# ---------------------------
_original_run = _subprocess.run

# ---------------------------
# SAFE subprocess.run (GLOBAL PATCH)
# ---------------------------
def _safe_run(*args, **kwargs):
    """
    Forces UTF-8 decoding for all subprocess.run calls.
    Prevents UnicodeDecodeError on Windows.
    """

    # Apply ONLY if text mode is used
    if kwargs.get("text", False) or kwargs.get("universal_newlines", False):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "ignore")

    return _original_run(*args, **kwargs)

# ---------------------------
# APPLY PATCH GLOBALLY
# ---------------------------
_subprocess.run = _safe_run


# ---------------------------
# OPTIONAL: SAFE OLLAMA CALL
# (You already imported this)
# ---------------------------
def call_ollama_safe(prompt, model="llama3"):
    try:
        result = _subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Ollama error: {e}"
