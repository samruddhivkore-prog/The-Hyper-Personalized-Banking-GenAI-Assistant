"""Strip Markdown formatting from text before it ever reaches a customer.

Chunking strips this at index time (see app/rag/chunking.py) so stored chunks
are already clean; this is also applied defensively to any final answer
right before it's returned, in case a real LLM echoes formatting despite
instructions not to.

Headings are dropped entirely, not merged into the surrounding prose: a
document's title is already carried as separate `source_title` metadata, and
keeping body headings as plain-text fragments is what used to glue a
heading onto the next sentence (e.g. "Wire Transfer Fees A wire transfer is
..."). Every surviving line is forced to end with sentence punctuation
before lines are joined, so whitespace collapse never fuses two clauses that
had no punctuation between them in the original markdown.
"""
import re


def strip_markdown(text: str) -> str:
    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue  # heading — dropped, not merged; title is separate metadata
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"\*([^*]+)\*", r"\1", line)
        if not re.search(r"[.!?:]$", line):
            line += "."
        kept.append(line)
    return " ".join(kept)
