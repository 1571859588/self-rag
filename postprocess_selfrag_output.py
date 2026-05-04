#!/usr/bin/env python3
"""
Post-process Self-RAG raw output into MAEDA evaluator input format.

Self-RAG raw output (run_short_form.py):
  {
    "preds": [str, ...],
    "prompts": [str, ...],
    "metric_results": [...],
    "all_results": [dict, ...],
    "golds": [...],
    "metric": "match",
    "metric_mean": float,
    "scores": [...]
  }

MAEDA evaluator input (regression_test.py):
  Each item must have:
    - query_id
    - question
    - input_answer
    - input_analysis (optional, empty string if N/A)
    - gt_answer
    - gt_answer_points
    - reranked_knowledge
    - background_knowledge
    - error_type (list from "label" field)
    - output: {} (empty dict, evaluator fills it)
"""

import json
import argparse
import os

CONTROL_TOKENS = [
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]


def clean_selfrag_output(text: str) -> str:
    """Remove Self-RAG reflection tokens and clean up the answer."""
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")
    text = text.replace("</s>", "")
    text = text.strip()
    if text and text[0] == " ":
        text = text[1:]
    return text


def postprocess(raw_results_path: str, gt_data_path: str, output_path: str):
    with open(raw_results_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    with open(gt_data_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    preds = raw.get("preds", [])
    all_results = raw.get("all_results", [])

    assert len(preds) == len(gt_data), \
        f"Mismatch: {len(preds)} preds vs {len(gt_data)} GT items"

    eval_items = []
    for i, (pred, gt) in enumerate(zip(preds, gt_data)):
        cleaned_answer = clean_selfrag_output(pred)

        do_retrieve = False
        retrieved_ctxs = []
        if i < len(all_results) and isinstance(all_results[i], dict):
            for key, val in all_results[i].items():
                if key.startswith("retrieval_") and isinstance(val, dict):
                    do_retrieve = True
                    if "ctx" in val:
                        retrieved_ctxs.append(val["ctx"])

        eval_item = {
            "query_id": gt["query_id"],
            "question": gt["question"],
            "answer": cleaned_answer,
            "input_answer": cleaned_answer,
            "input_analysis": "",
            "gt_answer": gt["gt_answer"],
            "gt_answer_points": gt["gt_answer_points"],
            "reranked_knowledge": gt["reranked_knowledge"],
            "background_knowledge": gt.get("background_knowledge", ""),
            "error_type": gt.get("error_type", []),
            "output": {},
            "selfrag_raw_pred": pred,
            "selfrag_do_retrieve": do_retrieve,
            "selfrag_retrieved_ctxs": retrieved_ctxs,
        }
        eval_items.append(eval_item)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_items, f, indent=2, ensure_ascii=False)

    print(f"Post-processed {len(eval_items)} items -> {output_path}")

    n_retrieved = sum(1 for item in eval_items if item["selfrag_do_retrieve"])
    n_empty = sum(1 for item in eval_items if not item["input_answer"].strip())
    print(f"  Retrieved: {n_retrieved}/{len(eval_items)}")
    print(f"  Empty answers: {n_empty}/{len(eval_items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_results", type=str, required=True,
                        help="Path to Self-RAG raw results JSON")
    parser.add_argument("--gt_data", type=str, required=True,
                        help="Path to selfrag_gt.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for MAEDA evaluator input JSON")
    args = parser.parse_args()
    postprocess(args.raw_results, args.gt_data, args.output)
