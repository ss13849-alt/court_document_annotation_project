#!/usr/bin/env python3
"""Annotate the 20 collected New York Court of Appeals documents via OpenRouter.

Setup:
  pip install requests
  export OPENROUTER_API_KEY='your-key'  # do not put the key in this file
  python annotate_openrouter.py

Optional:
  export OPENROUTER_MODEL='openai/gpt-4.1-mini'
  python annotate_openrouter.py --input-dir /path/to/documents --metadata /path/to/metadata.csv

Writes annotated_dataset.jsonl (one JSON object per document) and
annotation_errors.jsonl. Successful rows are saved incrementally and reused on
subsequent runs unless --retry-all is supplied.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = """You are annotating court decisions for a research dataset. Use only the document text and metadata provided. Do not use outside knowledge. If the document does not establish a fact, use null or 'unclear' rather than guessing. Distinguish the court's disposition from which party may have prevailed below. Return valid JSON only, matching the requested schema. Keep quotations short and provide pinpoint page/paragraph references only if the supplied text contains them; otherwise use null."""

ANNOTATION_PROMPT = """Annotate this document for litigation outcome and decision type. Return exactly one JSON object with these keys:
{{
  "document_type": "majority_opinion | memorandum_decision | procedural_order | other | unclear",
  "litigants": [{"name": string, "role": "appellant | respondent | petitioner | defendant | plaintiff | other | unclear"}],
  "principal_legal_issue": string,
  "court_disposition": "affirmed | reversed | modified | vacated | dismissed | remitted | questions_certified | reargument_ordered | other | unclear",
  "disposition_details": string,
  "prevailing_party_at_this_stage": string or null,
  "reasoning_summary": string,
  "confidence": "high | medium | low",
  "uncertainties": [string]
}}

Rules:
- Identify the New York Court of Appeals' disposition in this document, not a lower court's result unless needed to explain the procedural posture.
- The party that wins the appeal may differ from the party that won below. If the document does not clearly identify the prevailing party at this stage, use null.
- If the document is a short procedural order or does not resolve the merits, classify it accordingly and explain that in the summary.
- Use concise, neutral language. Do not infer facts or legal issues that are not supported by the text.

Metadata:
CASE NAME: {case_name}
COURT: {court}
YEAR: {year}
SOURCE URL: {source_url}

DOCUMENT TEXT:
{document_text}
"""


def load_existing(path):
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("filename") and row.get("status") == "success":
                        rows[row["filename"]] = row
                except json.JSONDecodeError:
                    continue
    return rows


def call_openrouter(api_key, model, prompt, timeout, max_retries):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai/",
        "X-Title": "Court Document Annotation Assignment",
    }
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429:
                detail = response.text[:1000]
                retry_after = response.headers.get("Retry-After")
                if attempt == max_retries:
                    raise RuntimeError(
                        "OpenRouter returned HTTP 429 (rate limit / provider capacity). "
                        f"Retry-After={retry_after!r}; response={detail}"
                    )
                try:
                    wait = max(5, min(120, int(retry_after))) if retry_after else min(60, 15 * (attempt + 1))
                except ValueError:
                    wait = min(60, 15 * (attempt + 1))
                print(f"HTTP 429; waiting {wait}s before retry. Detail: {detail}", file=sys.stderr)
                time.sleep(wait)
                continue
            if response.status_code in (500, 502, 503, 504):
                if attempt == max_retries:
                    response.raise_for_status()
                wait = min(60, 5 * (attempt + 1))
                print(f"HTTP {response.status_code}; retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if not response.ok:
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            try:
                annotation = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Model response was not valid JSON: {content[:500]}") from exc
            return annotation, data.get("id"), data.get("usage")
        except (requests.RequestException, ValueError, KeyError, IndexError):
            if attempt == max_retries:
                raise
            wait = min(30, 5 * (attempt + 1))
            print(f"Request/parse error; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--metadata", type=Path, default=Path(__file__).resolve().parent / "metadata.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "annotated_dataset.jsonl")
    parser.add_argument("--errors", type=Path, default=Path(__file__).resolve().parent / "annotation_errors.jsonl")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-all", action="store_true", help="Re-annotate even documents already successful in output")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY environment variable. Set it locally; do not paste it into chat or save it in this script.")
    if not args.metadata.exists():
        sys.exit(f"Metadata CSV not found: {args.metadata}")

    with args.metadata.open(newline="", encoding="utf-8-sig") as f:
        documents = list(csv.DictReader(f))
    if not documents:
        sys.exit("Metadata CSV contains no rows.")

    completed = {} if args.retry_all else load_existing(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0

    for i, meta in enumerate(documents, start=1):
        filename = (meta.get("filename") or "").strip()
        if not filename:
            print(f"Skipping row {i}: missing filename", file=sys.stderr)
            failures += 1
            continue
        if filename in completed:
            print(f"[{i}/{len(documents)}] Already annotated: {filename}")
            continue
        path = args.input_dir / filename
        if not path.exists():
            error = {"filename": filename, "case_name": meta.get("case_name"), "status": "error", "error": f"Document not found: {path}"}
            with args.errors.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")
            failures += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            error = {"filename": filename, "case_name": meta.get("case_name"), "status": "error", "error": "Document text is empty"}
            with args.errors.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")
            failures += 1
            continue
        # Replace only named placeholders. str.format() would interpret JSON
        # braces in the annotation schema as formatting fields.
        prompt = ANNOTATION_PROMPT
        for placeholder, value in (
            ("{case_name}", meta.get("case_name", "Unknown")),
            ("{court}", meta.get("court", "Unknown")),
            ("{year}", meta.get("year", "Unknown")),
            ("{source_url}", meta.get("source_url", "")),
            ("{document_text}", text),
        ):
            prompt = prompt.replace(placeholder, str(value))
        print(f"[{i}/{len(documents)}] Annotating: {meta.get('case_name', filename)}")
        try:
            annotation, request_id, usage = call_openrouter(api_key, args.model, prompt, args.timeout, args.max_retries)
            row = {
                **meta,
                "status": "success",
                "model": args.model,
                "openrouter_request_id": request_id,
                "usage": usage,
                "annotation": annotation,
            }
            completed[filename] = row
            # Rewrite the JSONL atomically after each success to preserve progress.
            tmp = args.output.with_suffix(args.output.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for item in completed.values():
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp.replace(args.output)
            successes += 1
        except Exception as exc:
            error = {"filename": filename, "case_name": meta.get("case_name"), "status": "error", "model": args.model, "error": str(exc)}
            with args.errors.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error, ensure_ascii=False) + "\n")
            print(f"  ERROR: {exc}", file=sys.stderr)
            failures += 1
        time.sleep(10)

    print(f"Finished. Newly annotated: {successes}; failures: {failures}; successful records saved: {len(completed)}")
    print(f"Dataset: {args.output}")
    if failures:
        print(f"Errors: {args.errors}")

if __name__ == "__main__":
    main()
