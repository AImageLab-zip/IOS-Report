import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from metrics import FieldExtractor


def load_ground_truth(captions_dir: Path) -> Dict[str, str]:
    ground_truth: Dict[str, str] = {}

    for json_file in captions_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            candidate = None

            if isinstance(data, dict):
                if "ios" in data and isinstance(data["ios"], dict):
                    candidate = data["ios"]

                if candidate is None:
                    for v in data.values():
                        if isinstance(v, dict) and "patient_id" in v and (
                            "description" in v or "original_italian" in v
                        ):
                            candidate = v
                            break

                if candidate is None and "patient_id" in data and (
                    "description" in data or "original_italian" in data
                ):
                    candidate = data

            if candidate is not None:
                patient_id = candidate.get("patient_id")
                description = candidate.get("description") or candidate.get("original_italian")
                if patient_id and description:
                    ground_truth[str(patient_id)] = str(description)
        except Exception:
            continue

    return ground_truth


def load_model_predictions(model_dir: Path) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    for txt_file in model_dir.glob("*.txt"):
        try:
            predictions[txt_file.stem] = txt_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return predictions


def align_predictions_and_references(
    predictions: Dict[str, str],
    references: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    common_ids = sorted(set(predictions.keys()) & set(references.keys()))
    ids = []
    pred_list = []
    ref_list = []
    for pid in common_ids:
        ids.append(pid)
        pred_list.append(predictions[pid])
        ref_list.append(references[pid])
    return ids, pred_list, ref_list


def load_patient_reports(captions_dir: Path) -> Dict[str, List[Dict]]:
    patients: Dict[str, List[Dict]] = {}
    for json_file in sorted(captions_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict) or not data:
            continue

        entries = []
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            desc = value.get("description")
            ann = value.get("annotator_id")
            vcid = value.get("voice_caption_id")
            if not isinstance(desc, str) or not desc.strip() or ann is None:
                continue
            entries.append(
                {
                    "report_key": str(key),
                    "patient_id": str(value.get("patient_id", json_file.stem)),
                    "annotator_id": str(ann),
                    "voice_caption_id": None if vcid is None else str(vcid),
                    "description": desc.strip(),
                }
            )

        if entries:
            patients[json_file.stem] = entries

    return patients


def build_pairs(entries: List[Dict], same_annotator: bool) -> List[Tuple[Dict, Dict]]:
    pairs: List[Tuple[Dict, Dict]] = []
    for a, b in combinations(entries, 2):
        same = a["annotator_id"] == b["annotator_id"]
        if same_annotator and same:
            va = a.get("voice_caption_id")
            vb = b.get("voice_caption_id")
            if va is not None and vb is not None and va == vb:
                continue
            pairs.append((a, b))
        elif (not same_annotator) and (not same):
            pairs.append((a, b))
    return pairs


def make_directional_rows(a: Dict, b: Dict) -> List[Dict]:
    return [
        {"pred_text": a["description"], "ref_text": b["description"]},
        {"pred_text": b["description"], "ref_text": a["description"]},
    ]


def compute_per_field_accuracy(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    field_scores: Dict[str, List[float]] = {k: [] for k in FieldExtractor.FIELD_PATTERNS.keys()}

    for pred, ref in zip(predictions, references):
        pred_fields = FieldExtractor.extract_fields(pred)
        ref_fields = FieldExtractor.extract_fields(ref)

        if not ref_fields:
            continue

        ref_fields_filtered = {
            k: v for k, v in ref_fields.items() if not FieldExtractor.should_skip_field(k, v)
        }
        pred_fields_filtered = {
            k: v for k, v in pred_fields.items() if not FieldExtractor.should_skip_field(k, v)
        }

        for field_name, ref_value in ref_fields_filtered.items():
            if field_name in pred_fields_filtered:
                score = FieldExtractor.compare_values(
                    pred_fields_filtered[field_name],
                    ref_value,
                    field_type=field_name,
                )
            else:
                score = 0.0
            field_scores.setdefault(field_name, []).append(float(score))

    out: Dict[str, Dict[str, float]] = {}
    for field_name in FieldExtractor.FIELD_PATTERNS.keys():
        scores = field_scores.get(field_name, [])
        out[field_name] = {
            "accuracy": float(np.mean(scores)) if scores else 0.0,
            "n": int(len(scores)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-field accuracy for models and inter/intra annotator agreement"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="accuracy_per_field.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory containing model prediction folders",
    )
    parser.add_argument(
        "--model_captions_dir",
        type=str,
        default="_data/captions",
        help="Ground-truth captions directory for model evaluation",
    )
    parser.add_argument(
        "--annotator_captions_dir",
        type=str,
        default=None,
        help="Captions directory for intra/inter annotator evaluation",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of model folder names to evaluate",
    )
    args = parser.parse_args()

    rows: List[Dict] = []

    output_dir = Path(args.output_dir)
    model_captions_dir = Path(args.model_captions_dir)

    if output_dir.exists() and model_captions_dir.exists():
        ground_truth = load_ground_truth(model_captions_dir)

        if args.models:
            model_dirs = [output_dir / model_name for model_name in args.models]
            model_dirs = [d for d in model_dirs if d.exists() and d.is_dir()]
        else:
            model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

        for model_dir in sorted(model_dirs, key=lambda p: p.name):
            preds = load_model_predictions(model_dir)
            _, pred_list, ref_list = align_predictions_and_references(preds, ground_truth)
            if not pred_list:
                continue

            per_field = compute_per_field_accuracy(pred_list, ref_list)
            for field_name, stats in per_field.items():
                rows.append(
                    {
                        "group": "model",
                        "entity": model_dir.name,
                        "field": field_name,
                        "accuracy": stats["accuracy"],
                        "n": stats["n"],
                    }
                )

    if args.annotator_captions_dir:
        annotator_dir = Path(args.annotator_captions_dir)
        if annotator_dir.exists():
            patients = load_patient_reports(annotator_dir)
            inter_preds: List[str] = []
            inter_refs: List[str] = []
            intra_preds: List[str] = []
            intra_refs: List[str] = []

            for entries in patients.values():
                inter_pairs = build_pairs(entries, same_annotator=False)
                intra_pairs = build_pairs(entries, same_annotator=True)

                for a, b in inter_pairs:
                    for row in make_directional_rows(a, b):
                        inter_preds.append(row["pred_text"])
                        inter_refs.append(row["ref_text"])

                for a, b in intra_pairs:
                    for row in make_directional_rows(a, b):
                        intra_preds.append(row["pred_text"])
                        intra_refs.append(row["ref_text"])

            for entity, preds, refs in (
                ("inter_annotator", inter_preds, inter_refs),
                ("intra_annotator", intra_preds, intra_refs),
            ):
                if not preds:
                    continue
                per_field = compute_per_field_accuracy(preds, refs)
                for field_name, stats in per_field.items():
                    rows.append(
                        {
                            "group": "annotator",
                            "entity": entity,
                            "field": field_name,
                            "accuracy": stats["accuracy"],
                            "n": stats["n"],
                        }
                    )

    if not rows:
        raise RuntimeError("No evaluable samples found. Check input directories and formats.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "entity", "field"]).reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, float_format="%.6f")

    print(f"Saved per-field accuracy table to: {output_csv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
