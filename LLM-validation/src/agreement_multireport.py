import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from metrics import compute_bleu, compute_field_accuracy, compute_meteor, compute_radfact_per_sample, compute_rouge


def load_patient_reports(captions_dir: Path) -> Dict[str, List[Dict]]:
    patients = {}
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
    pairs = []
    for a, b in combinations(entries, 2):
        same = a["annotator_id"] == b["annotator_id"]
        if same_annotator and same:
            # If voice caption IDs exist, enforce difference when available
            va = a.get("voice_caption_id")
            vb = b.get("voice_caption_id")
            if va is not None and vb is not None and va == vb:
                continue
            pairs.append((a, b))
        elif (not same_annotator) and (not same):
            pairs.append((a, b))
    return pairs


def make_directional_rows(entries_a: Dict, entries_b: Dict) -> List[Dict]:
    return [
        {
            "patient_id": entries_a["patient_id"],
            "pred_key": entries_a["report_key"],
            "ref_key": entries_b["report_key"],
            "pred_annotator_id": entries_a["annotator_id"],
            "ref_annotator_id": entries_b["annotator_id"],
            "pred_voice_caption_id": entries_a.get("voice_caption_id"),
            "ref_voice_caption_id": entries_b.get("voice_caption_id"),
            "pred_text": entries_a["description"],
            "ref_text": entries_b["description"],
        },
        {
            "patient_id": entries_b["patient_id"],
            "pred_key": entries_b["report_key"],
            "ref_key": entries_a["report_key"],
            "pred_annotator_id": entries_b["annotator_id"],
            "ref_annotator_id": entries_a["annotator_id"],
            "pred_voice_caption_id": entries_b.get("voice_caption_id"),
            "ref_voice_caption_id": entries_a.get("voice_caption_id"),
            "pred_text": entries_b["description"],
            "ref_text": entries_a["description"],
        },
    ]


def evaluate_rows_fast(rows: List[Dict], include_sbert: bool, sbert_device: str) -> None:
    for row in tqdm(rows, desc="Lexical metrics", unit="sample"):
        pred = row["pred_text"]
        ref = row["ref_text"]

        row.update(compute_field_accuracy([pred], [ref], return_per_field=False))
        row.update(compute_bleu([pred], [ref], n_gram=1))
        row.update(compute_bleu([pred], [ref], n_gram=4))
        row.update(compute_rouge([pred], [ref]))
        row.update(compute_meteor([pred], [ref]))

    if include_sbert and rows:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2", device=sbert_device)
        preds = [r["pred_text"] for r in rows]
        refs = [r["ref_text"] for r in rows]
        pred_emb = model.encode(preds, convert_to_tensor=True, device=sbert_device, batch_size=64)
        ref_emb = model.encode(refs, convert_to_tensor=True, device=sbert_device, batch_size=64)
        sims = util.cos_sim(pred_emb, ref_emb).diagonal().detach().cpu().numpy()
        for row, score in zip(rows, sims):
            row["sbert-sim"] = float(score)


def summarize(rows: List[Dict], metric_keys: List[str]) -> Dict:
    out = {
        "n_directional_samples": len(rows),
        "metrics": {},
    }

    for key in metric_keys:
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
        if not vals:
            out["metrics"][key] = {
                "mean": 0.0,
                "variance": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "n": 0,
            }
            continue

        arr = np.array(vals, dtype=float)
        var = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
        std = float(np.sqrt(var))
        out["metrics"][key] = {
            "mean": float(np.mean(arr)),
            "variance": var,
            "std": std,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    return out


def bootstrap_ci_mean(values: np.ndarray, iters: int, rng: np.random.Generator) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        v = float(values[0])
        return v, v

    n = values.size
    boot_means = np.empty(iters, dtype=float)
    for i in range(iters):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return low, high


def summarize_with_bootstrap(
    rows: List[Dict], metric_keys: List[str], bootstrap_iters: int, bootstrap_seed: int
) -> Dict:
    out = {
        "n_directional_samples": len(rows),
        "bootstrap": {
            "iters": bootstrap_iters,
            "seed": bootstrap_seed,
            "ci": "percentile_95",
            "unit": "directional_samples",
        },
        "metrics": {},
    }

    rng = np.random.default_rng(bootstrap_seed)

    for key in metric_keys:
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
        if not vals:
            out["metrics"][key] = {
                "mean": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "variance": 0.0,
                "std": 0.0,
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "min": 0.0,
                "max": 0.0,
                "n": 0,
            }
            continue

        arr = np.array(vals, dtype=float)
        var = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
        std = float(np.sqrt(var))
        ci_low, ci_high = bootstrap_ci_mean(arr, bootstrap_iters, rng)

        out["metrics"][key] = {
            "mean": float(np.mean(arr)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "variance": var,
            "std": std,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Compute inter/intra annotator agreement on multireport captions")
    parser.add_argument("--captions_dir", type=str, required=True, help="Directory with patient JSON caption files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save summary JSON")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save per-direction metrics CSV")
    parser.add_argument("--no-sbert", action="store_true", help="Skip Sentence-BERT metric")
    parser.add_argument("--sbert-device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--radfact", action="store_true", help="Include RadFact metrics")
    parser.add_argument("--radfact-model", type=str, default="gpt-5-nano")
    parser.add_argument("--radfact-rate-limit", type=int, default=400)
    parser.add_argument("--radfact-workers", type=int, default=5)
    parser.add_argument("--radfact-cache-file", type=str, default=None, help="Path to RadFact per-sample cache JSON")
    parser.add_argument("--ollama-url", type=str, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=5000, help="Bootstrap iterations for 95% CI")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Random seed for bootstrap CI")
    args = parser.parse_args()

    captions_dir = Path(args.captions_dir)
    if not captions_dir.exists():
        raise FileNotFoundError(f"Captions directory not found: {captions_dir}")

    patients = load_patient_reports(captions_dir)
    print(f"Loaded {len(patients)} patients from {captions_dir}")

    inter_rows: List[Dict] = []
    intra_rows: List[Dict] = []
    patients_with_inter = 0
    patients_with_intra = 0
    inter_pair_count = 0
    intra_pair_count = 0

    # Pre-scan pairs for progress visibility
    patient_pair_plan = {}
    for patient_key, entries in patients.items():
        inter_pairs = build_pairs(entries, same_annotator=False)
        intra_pairs = build_pairs(entries, same_annotator=True)
        patient_pair_plan[patient_key] = (entries, inter_pairs, intra_pairs)
        inter_pair_count += len(inter_pairs)
        intra_pair_count += len(intra_pairs)

    total_directional = 2 * (inter_pair_count + intra_pair_count)
    print(f"Planned inter pairs: {inter_pair_count}")
    print(f"Planned intra pairs: {intra_pair_count}")
    print(f"Planned directional evaluations: {total_directional}")

    for patient_key, (entries, inter_pairs, intra_pairs) in tqdm(
        patient_pair_plan.items(), desc="Patients", unit="patient"
    ):

        if inter_pairs:
            patients_with_inter += 1
        if intra_pairs:
            patients_with_intra += 1

        for pair in inter_pairs:
            a, b = pair
            inter_rows.extend(make_directional_rows(a, b))

        for pair in intra_pairs:
            a, b = pair
            intra_rows.extend(make_directional_rows(a, b))

    print(f"Built inter directional rows: {len(inter_rows)}")
    print(f"Built intra directional rows: {len(intra_rows)}")

    print("Computing inter lexical/SBERT metrics...")
    evaluate_rows_fast(inter_rows, include_sbert=not args.no_sbert, sbert_device=args.sbert_device)
    print("Computing intra lexical/SBERT metrics...")
    evaluate_rows_fast(intra_rows, include_sbert=not args.no_sbert, sbert_device=args.sbert_device)

    if args.radfact:
        print("Computing inter RadFact metrics per sample...")
        inter_pred = [r["pred_text"] for r in inter_rows]
        inter_ref = [r["ref_text"] for r in inter_rows]
        inter_rad = compute_radfact_per_sample(
            inter_pred,
            inter_ref,
            radfact_model=args.radfact_model,
            cache_file=args.radfact_cache_file,
            ollama_url=args.ollama_url,
            max_workers=args.radfact_workers,
            rate_limit_rpm=args.radfact_rate_limit,
        )
        print("Computing intra RadFact metrics per sample...")
        intra_pred = [r["pred_text"] for r in intra_rows]
        intra_ref = [r["ref_text"] for r in intra_rows]
        intra_rad = compute_radfact_per_sample(
            intra_pred,
            intra_ref,
            radfact_model=args.radfact_model,
            cache_file=args.radfact_cache_file,
            ollama_url=args.ollama_url,
            max_workers=args.radfact_workers,
            rate_limit_rpm=args.radfact_rate_limit,
        )
        for r, m in zip(inter_rows, inter_rad):
            r.update(m)
        for r, m in zip(intra_rows, intra_rad):
            r.update(m)

    metric_keys = [
        "accuracy",
        "coverage",
        "num_fields",
        "bleu-1",
        "bleu-4",
        "rouge-l-p",
        "rouge-l-r",
        "rouge-l-f",
        "meteor",
    ]
    if not args.no_sbert:
        metric_keys.append("sbert-sim")
    if args.radfact:
        metric_keys.extend(["radfact-precision", "radfact-recall", "radfact-f1"])

    inter_summary = summarize_with_bootstrap(inter_rows, metric_keys, args.bootstrap_iters, args.bootstrap_seed)
    intra_summary = summarize_with_bootstrap(intra_rows, metric_keys, args.bootstrap_iters, args.bootstrap_seed)

    flat_table_rows = []
    for agreement_type, section in (("inter", inter_summary), ("intra", intra_summary)):
        for metric_name, stats in section["metrics"].items():
            flat_table_rows.append(
                {
                    "agreement_type": agreement_type,
                    "metric": metric_name,
                    "mean": stats["mean"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "std": stats["std"],
                    "variance": stats["variance"],
                    "median": stats["median"],
                    "q25": stats["q25"],
                    "q75": stats["q75"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "n": stats["n"],
                }
            )

    summary = {
        "config": {
            "captions_dir": str(captions_dir),
            "include_sbert": not args.no_sbert,
            "sbert_device": args.sbert_device,
            "include_radfact": args.radfact,
            "radfact_model": args.radfact_model,
            "radfact_rate_limit": args.radfact_rate_limit,
            "radfact_workers": args.radfact_workers,
            "radfact_cache_file": args.radfact_cache_file,
            "ollama_url": args.ollama_url,
            "variance": "sample_variance_ddof_1",
            "bootstrap_ci": "nonparametric_percentile_95_over_directional_samples",
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "pairing": "all_valid_pairs_bidirectional",
        },
        "counts": {
            "patients_total": len(patients),
            "patients_with_inter_pairs": patients_with_inter,
            "patients_with_intra_pairs": patients_with_intra,
            "inter_undirected_pairs": inter_pair_count,
            "intra_undirected_pairs": intra_pair_count,
            "inter_directional_samples": len(inter_rows),
            "intra_directional_samples": len(intra_rows),
        },
        "inter_annotator": inter_summary,
        "intra_annotator": intra_summary,
        "paper_table": flat_table_rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in inter_rows:
            item = dict(r)
            item["agreement_type"] = "inter"
            rows.append(item)
        for r in intra_rows:
            item = dict(r)
            item["agreement_type"] = "intra"
            rows.append(item)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("Agreement analysis complete")
    print(f"- Inter directional samples: {len(inter_rows)}")
    print(f"- Intra directional samples: {len(intra_rows)}")
    print(f"- Summary saved to: {output_json}")
    if args.output_csv:
        print(f"- Per-direction CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
