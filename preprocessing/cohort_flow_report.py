#!/usr/bin/env python3
"""Generate cohort flow summary statistics for staged cohort CSV files.

Outputs per stage:
- total_stays (unique ICU stays)
- total_patients (unique patients, if patient id available)
- age_median
- female_pct

The script is robust to trajectory tables with multiple rows per stay by
computing demographics on one representative row per ICU stay.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple


PREFERRED_PATIENT_COLS = ["subject_id", "patient_id", "subjectid"]
PREFERRED_STAY_COLS = ["icustayid", "icustay_id", "stay_id"]
PREFERRED_AGE_COLS = ["age", "Age"]
PREFERRED_GENDER_COLS = ["gender", "sex", "Gender", "Sex"]


@dataclass
class StageSummary:
    stage: str
    total_stays: str
    total_patients: str
    age_median: str
    female_pct: str
    note: str = ""


def _pick_col(headers: Iterable[str], preferred: List[str]) -> Optional[str]:
    header_set = set(headers)
    for col in preferred:
        if col in header_set:
            return col
    return None


def _parse_float(v: str) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _gender_to_female(v: str) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"f", "female"}:
        return 1
    if s in {"m", "male"}:
        return 0

    num = _parse_float(s)
    if num is None:
        return None
    # Common encoding in this repo: 1=female, 0=male.
    if num in {0.0, 1.0}:
        return int(num)
    # Alternate encoding seen in some exports: 2=female, 1=male.
    if num in {1.0, 2.0}:
        return 1 if int(num) == 2 else 0
    return None


def summarize_stage(
    stage_name: str,
    csv_path: Path,
    patient_map: Optional[Dict[str, str]] = None,
) -> StageSummary:
    if not csv_path.exists():
        return StageSummary(stage_name, "NA", "NA", "NA", "NA", f"missing file: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return StageSummary(stage_name, "NA", "NA", "NA", "NA", "empty CSV")

        headers = reader.fieldnames
        stay_col = _pick_col(headers, PREFERRED_STAY_COLS)
        patient_col = _pick_col(headers, PREFERRED_PATIENT_COLS)
        age_col = _pick_col(headers, PREFERRED_AGE_COLS)
        gender_col = _pick_col(headers, PREFERRED_GENDER_COLS)

        if stay_col is None:
            return StageSummary(stage_name, "NA", "NA", "NA", "NA", "no ICU stay column found")

        # Keep one representative row per stay (first occurrence).
        by_stay: Dict[str, Dict[str, str]] = {}
        for row in reader:
            sid = str(row.get(stay_col, "")).strip()
            if sid == "":
                continue
            if sid not in by_stay:
                by_stay[sid] = row

        stays = len(by_stay)
        ages: List[float] = []
        female_values: List[int] = []
        patients: set[str] = set()

        for sid, row in by_stay.items():
            if age_col:
                age = _parse_float(row.get(age_col, ""))
                if age is not None:
                    ages.append(age)
            if gender_col:
                gf = _gender_to_female(row.get(gender_col, ""))
                if gf is not None:
                    female_values.append(gf)

            pid: Optional[str] = None
            if patient_col:
                pid_raw = str(row.get(patient_col, "")).strip()
                if pid_raw:
                    pid = pid_raw
            elif patient_map is not None:
                pid = patient_map.get(sid)

            if pid:
                patients.add(pid)

        age_str = f"{median(ages):.2f}" if ages else "NA"
        female_str = f"{(sum(female_values) / len(female_values) * 100):.2f}" if female_values else "NA"

        patient_note = ""
        if patient_col is None and patient_map is None:
            patient_count = "NA"
            patient_note = "patient id unavailable"
        else:
            patient_count = str(len(patients))

        return StageSummary(
            stage=stage_name,
            total_stays=str(stays),
            total_patients=patient_count,
            age_median=age_str,
            female_pct=female_str,
            note=patient_note,
        )


def load_patient_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping
        stay_col = _pick_col(reader.fieldnames, PREFERRED_STAY_COLS)
        patient_col = _pick_col(reader.fieldnames, PREFERRED_PATIENT_COLS)
        if not stay_col or not patient_col:
            return mapping
        for row in reader:
            sid = str(row.get(stay_col, "")).strip()
            pid = str(row.get(patient_col, "")).strip()
            if sid and pid and sid not in mapping:
                mapping[sid] = pid
    return mapping


def write_outputs(rows: List[StageSummary], out_csv: Optional[Path], out_md: Optional[Path]) -> None:
    headers = ["stage", "total_stays", "total_patients", "age_median", "female_pct", "note"]

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow([r.stage, r.total_stays, r.total_patients, r.age_median, r.female_pct, r.note])

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "| stage | total_stays | total_patients | age_median | female_pct | note |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r.stage} | {r.total_stays} | {r.total_patients} | {r.age_median} | {r.female_pct} | {r.note} |"
            )
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cohort flow summary report")

    # Optional explicit stage files.
    p.add_argument("--all-icu", type=Path, help="CSV for all ICU stays stage")
    p.add_argument("--sepsis-candidates", type=Path, help="CSV for sepsis-3 candidates stage")
    p.add_argument("--exclude-no-iv", type=Path, help="CSV after excluding no-IV-fluid stays")
    p.add_argument("--exclude-missing", type=Path, help="CSV after excluding >8 missing features at onset")
    p.add_argument("--final-train", type=Path, help="CSV for final train cohort")
    p.add_argument("--final-val", type=Path, help="CSV for final val cohort")
    p.add_argument("--final-test", type=Path, help="CSV for final test cohort")

    # Convenience: infer final splits from config-like defaults.
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Default directory for final split CSVs")

    p.add_argument(
        "--patient-map",
        type=Path,
        help="Optional CSV containing both ICU stay id and patient id for patient counting",
    )
    p.add_argument("--out-csv", type=Path, default=Path("results/cohort_flow_summary.csv"))
    p.add_argument("--out-md", type=Path, default=Path("results/cohort_flow_summary.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    patient_map = load_patient_map(args.patient_map) if args.patient_map else None

    final_train = args.final_train or (args.data_dir / "rl_train_data_final_cont.csv")
    final_val = args.final_val or (args.data_dir / "rl_val_data_final_cont.csv")
    final_test = args.final_test or (args.data_dir / "rl_test_data_final_cont.csv")

    stages: List[Tuple[str, Optional[Path]]] = [
        ("All ICU stays", args.all_icu),
        ("Sepsis-3 candidates", args.sepsis_candidates),
        ("After excluding no IV fluids", args.exclude_no_iv),
        ("After excluding >8 missing features at onset", args.exclude_missing),
    ]

    rows: List[StageSummary] = []
    for name, path in stages:
        if path is None:
            rows.append(StageSummary(name, "NA", "NA", "NA", "NA", "file not provided"))
        else:
            rows.append(summarize_stage(name, path, patient_map))

    # Final train+val from both files together (deduplicate by stay across splits).
    temp_combined = args.out_csv.parent / "_tmp_train_val_for_summary.csv"
    temp_combined.parent.mkdir(parents=True, exist_ok=True)
    with temp_combined.open("w", encoding="utf-8", newline="") as out_f:
        writer = None
        for p in [final_train, final_val]:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if not reader.fieldnames:
                    continue
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    rows.append(summarize_stage("Final cohort (train+val)", temp_combined, patient_map))
    rows.append(summarize_stage("Final cohort test", final_test, patient_map))

    write_outputs(rows, args.out_csv, args.out_md)

    if temp_combined.exists():
        temp_combined.unlink()

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")


if __name__ == "__main__":
    main()
