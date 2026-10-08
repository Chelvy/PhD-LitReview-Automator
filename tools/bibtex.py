"""
Deterministic BibTeX generation from paper metadata.

Entries are built from the metadata the search APIs returned (authors, title, year,
venue, DOI, arXiv ID) rather than written by the LLM, so DOIs and author lists are
never invented and every citation key is unique within the bibliography.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_KEY_STOPWORDS = {"a", "an", "the", "on", "of", "for", "in", "to", "and", "towards", "toward", "with"}

_ENTRY_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")


def escape_latex(text: Any) -> str:
    return "".join(_LATEX_SPECIAL.get(c, c) for c in str(text or ""))


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _surname(author: str) -> str:
    author = author.strip()
    if "," in author:
        return author.split(",")[0].strip()
    parts = author.split()
    return parts[-1] if parts else ""


def existing_keys(bib_text: str) -> set[str]:
    """Citation keys already present in a .bib file's text."""
    return set(_ENTRY_KEY_RE.findall(bib_text or ""))


def citation_key(paper: dict[str, Any], taken: set[str]) -> str:
    """Build a unique key like 'Smith2025Compositional' (suffixed a, b, ... on collision)."""
    authors = paper.get("authors") or []
    surname = re.sub(r"[^A-Za-z]", "", _ascii(_surname(str(authors[0])))) if authors else ""
    year = str(paper.get("year") or "nd")
    words = [w for w in re.findall(r"[A-Za-z]+", _ascii(str(paper.get("title") or ""))) if w.lower() not in _KEY_STOPWORDS]
    base = f"{surname or 'Anon'}{year}{words[0].capitalize() if words else ''}"

    key, n = base, 0
    while key in taken:
        key = f"{base}{chr(ord('a') + n)}" if n < 26 else f"{base}{n}"
        n += 1
    taken.add(key)
    return key


def _entry_type(paper: dict[str, Any]) -> tuple[str, str]:
    """Return (entry type, venue field name)."""
    venue = str(paper.get("venue") or "")
    if paper.get("arxiv_id") and (not venue or venue.lower().startswith("arxiv")):
        return "misc", ""
    if re.search(r"conference|proceedings|symposium|workshop|\bconf\b", venue, re.IGNORECASE):
        return "inproceedings", "booktitle"
    return "article", "journal"


def bibtex_entry(paper: dict[str, Any], key: str) -> str:
    """Render one BibTeX entry for a paper dict."""
    entry_type, venue_field = _entry_type(paper)
    fields: list[tuple[str, str]] = []

    authors = [str(a) for a in (paper.get("authors") or []) if a]
    if authors:
        fields.append(("author", " and ".join(escape_latex(a) for a in authors)))
    # Double braces keep the title's capitalisation in most bibliography styles
    fields.append(("title", "{" + escape_latex(paper.get("title") or "Untitled") + "}"))
    if paper.get("year"):
        fields.append(("year", str(paper["year"])))
    if venue_field and paper.get("venue"):
        fields.append((venue_field, escape_latex(paper["venue"])))
    if paper.get("arxiv_id"):
        fields.append(("eprint", str(paper["arxiv_id"])))
        fields.append(("archivePrefix", "arXiv"))
    if paper.get("doi"):
        fields.append(("doi", str(paper["doi"])))
    if paper.get("url"):
        fields.append(("url", str(paper["url"])))

    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def build_entries(papers: Iterable[dict[str, Any]], taken: set[str]) -> dict[str, tuple[str, str]]:
    """
    Build entries for papers that don't already carry a citation key.
    Returns {paper_id: (key, bibtex)}; `taken` is updated with the new keys.
    """
    entries: dict[str, tuple[str, str]] = {}
    for paper in papers:
        paper_id = paper.get("paper_id")
        if not paper_id or paper_id in entries:
            continue
        key = paper.get("citation_key")
        if key and key in taken and paper.get("bibtex"):
            continue  # already in the bibliography
        if not key or key in taken:
            key = citation_key(paper, taken)
        else:
            taken.add(key)  # keep a key the paper already has (e.g. cited in the documents)
        entries[paper_id] = (key, bibtex_entry(paper, key))
    return entries
