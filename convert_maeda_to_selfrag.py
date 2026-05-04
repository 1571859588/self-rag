#!/usr/bin/env python3
"""
Convert MAEDA-benchmark-300.json to Self-RAG input format.

MAEDA benchmark format:
  - query_id, question, answer, reranked_knowledge (raw text with "id:xxx\n### Title\n..."),
    gt_answer, gt_answer_points, gt_reference_doc_ids, label

Self-RAG input format (for run_short_form.py with pre-retrieved docs):
  - instruction (or question), ctxs: [{title, text}], answers

Output:
  1. selfrag_input.json  — for Self-RAG inference
  2. selfrag_gt.json     — for MAEDA evaluator (preserves all original fields)
"""

import json
import re
import argparse
import os


def parse_reranked_knowledge(rk_text: str) -> list:
    """Parse reranked_knowledge string into list of {title, text} dicts.

    Format of rk_text:
        id:doc_id_1
        ### Title 1
        content...

        id:doc_id_2
        ### Title 2
        content...
    """
    if not rk_text or not rk_text.strip():
        return []

    # Split by newline that precedes "id:" (but not at position 0)
    segments = re.split(r'\n(?=id:)', rk_text.strip())

    ctxs = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Extract doc_id from first line: "id:xxx"
        lines = seg.split('\n', 1)
        first_line = lines[0]
        doc_id = first_line.replace('id:', '', 1).strip()

        # Extract title after "###"
        title_match = re.search(r'###\s*(.+)', seg)
        if title_match:
            title = title_match.group(1).strip()
            # Text starts after the title line
            remaining = seg[title_match.end():]
        else:
            title = doc_id
            remaining = lines[1] if len(lines) > 1 else ""

        text = remaining.strip()

        ctxs.append({
            "title": title,
            "text": text,
            "id": doc_id,
        })

    return ctxs


def convert(input_path: str, output_dir: str, ndocs: int = 10):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    selfrag_data = []
    gt_data = []

    for item in benchmark_data:
        question = item["question"]
        rk_text = item.get("reranked_knowledge", "")
        ctxs = parse_reranked_knowledge(rk_text)
        # Limit to top-ndocs
        ctxs = ctxs[:ndocs]

        # Self-RAG format
        selfrag_item = {
            "instruction": question,
            "question": question,
            "ctxs": ctxs,
            "answers": [item.get("gt_answer", "")],
            "query_id": item.get("query_id", 0),
        }
        selfrag_data.append(selfrag_item)

        # GT data for MAEDA evaluator (preserved original fields)
        gt_item = {
            "query_id": item.get("query_id", 0),
            "question": question,
            "gt_answer": item.get("gt_answer", ""),
            "gt_answer_points": item.get("gt_answer_points", []),
            "reranked_knowledge": rk_text,
            "background_knowledge": item.get("background_knowledge", ""),
            "error_type": item.get("label", []),
            "gt_reference_doc_ids": item.get("gt_reference_doc_ids", []),
        }
        gt_data.append(gt_item)

    # Save
    selfrag_path = os.path.join(output_dir, "selfrag_input.json")
    gt_path = os.path.join(output_dir, "selfrag_gt.json")

    with open(selfrag_path, 'w', encoding='utf-8') as f:
        json.dump(selfrag_data, f, indent=2, ensure_ascii=False)

    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(benchmark_data)} items")
    print(f"  Self-RAG input: {selfrag_path}")
    print(f"  GT data:        {gt_path}")

    # Stats
    n_empty_ctxs = sum(1 for d in selfrag_data if len(d["ctxs"]) == 0)
    n_ctxs = [len(d["ctxs"]) for d in selfrag_data]
    print(f"  Empty ctxs: {n_empty_ctxs}, avg ctxs: {sum(n_ctxs)/len(n_ctxs):.1f}, max: {max(n_ctxs)}, min: {min(n_ctxs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="../../MAEDA-benchmark-300.json",
                        # self-rag/ → baselines/ → MAEDA-DATE26/
                        help="Path to MAEDA-benchmark-300.json")
    parser.add_argument("--output_dir", type=str,
                        default="./maeda_selfrag_data",
                        help="Output directory")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of docs per question to include")
    args = parser.parse_args()
    convert(args.input, args.output_dir, args.ndocs)
