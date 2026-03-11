"""
rag_safe.py
-------------------------------------------------
DROP-IN RAG PATCH (PDF / CSV)
- No logic changes required
- Automatically injects document context into LLM
-------------------------------------------------
"""

import subprocess
import os
import re
import csv

DOCUMENT_CACHE = ""

def _load_documents():
    global DOCUMENT_CACHE
    texts = []

    # CSV
    for file in os.listdir("."):
        if file.lower().endswith(".csv"):
            try:
                with open(file, encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    texts.append(
                        f"CSV FILE: {file}\n" +
                        "\n".join([" | ".join(r) for r in rows[:50]])
                    )
            except:
                pass

    # PDF
    try:
        import PyPDF2
        for file in os.listdir("."):
            if file.lower().endswith(".pdf"):
                with open(file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = [p.extract_text() for p in reader.pages[:5]]
                    texts.append(f"PDF FILE: {file}\n" + "\n".join(pages))
    except:
        pass

    DOCUMENT_CACHE = "\n\n".join(texts)


def _retrieve_context(query, limit=1500):
    if not DOCUMENT_CACHE:
        return ""

    words = set(re.findall(r"\w+", query.lower()))
    chunks = DOCUMENT_CACHE.split("\n")

    scored = []
    for c in chunks:
        score = sum(1 for w in words if w in c.lower())
        if score > 0:
            scored.append((score, c))

    scored.sort(reverse=True)
    return "\n".join([c for _, c in scored[:20]])[:limit]


_original_run = subprocess.run

def _rag_run(*args, **kwargs):
    if kwargs.get("text") and isinstance(kwargs.get("input"), str):
        query = kwargs["input"]

        if not DOCUMENT_CACHE:
            _load_documents()

        context = _retrieve_context(query)
        if context:
            kwargs["input"] = f"""
Answer ONLY using the context below.
If not found, say "Not found in document".

CONTEXT:
{context}

QUESTION:
{query}
"""

    return _original_run(*args, **kwargs)

subprocess.run = _rag_run
