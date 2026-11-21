"""Read Mulan/MEKA ARFF benchmarks without fitting preprocessing on future rows.

Uses the publicly declared ARFF schema, not future examples, to one-hot encode
categorical variables. Requires an explicit XML or MEKA -C label declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import csv
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class BenchmarkData:
    name: str
    records: list[tuple[tuple[float, ...], tuple[int, ...]]]
    raw_features: int
    encoded_features: int
    n_labels: int
    label_names: tuple[str, ...]
    arff_sha256: str
    encoding: str


PAPER_DATASETS = {
    # Esteban et al. (2024), paper Table 3 and Table 6.
    "flags": {"n": 194, "features": 19, "labels": 7, "paper_example_f1": 0.605},
    "emotions": {"n": 593, "features": 72, "labels": 6, "paper_example_f1": 0.446},
    "yeast": {"n": 2417, "features": 103, "labels": 14, "paper_example_f1": 0.588},
}


def _read_arff_stdlib(path: Path):
    """Small dense-ARFF fallback for Python environments lacking liac-arff.

    This explicitly rejects sparse/advanced ARFF syntax instead of misparsing.
    """
    relation = ""
    attributes = []
    data = []
    in_data = False
    pattern = re.compile(r"^@attribute\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))\s+(.+)$", re.I)
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('%'):
            continue
        if not in_data:
            if line.lower().startswith('@relation'):
                relation = line.split(None, 1)[1].strip().strip("'\"")
            elif line.lower().startswith('@attribute'):
                match = pattern.match(line)
                if match is None:
                    raise ValueError(f"Unsupported ARFF header at line {number}")
                name = next(v for v in match.groups()[:3] if v is not None)
                typ = match.group(4).strip()
                if typ.startswith('{') and typ.endswith('}'):
                    cats = next(csv.reader([typ[1:-1]], skipinitialspace=True))
                    typ = [x.strip().strip("'\"") for x in cats]
                attributes.append((name, typ))
            elif line.lower() == '@data':
                in_data = True
        else:
            if line.startswith('{'):
                raise ValueError('Sparse ARFF data requires liac-arff: python -m pip install liac-arff')
            cells = next(csv.reader([line], skipinitialspace=True))
            data.append([v.strip().strip("'\"") for v in cells])
    if not attributes or not in_data:
        raise ValueError('Invalid/incomplete ARFF: missing @attribute or @data')
    return {'relation': relation, 'attributes': attributes, 'data': data}


def _xml_label_names(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    names = [elem.attrib["name"] for elem in root.iter()
             if elem.tag.split("}")[-1].lower() == "label" and "name" in elem.attrib]
    if not names:
        raise ValueError(f"No <label name=...> entries in {xml_path}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate label names in XML")
    return names


def _extract_label_indices(relation: str, attributes, xml_path: Path | None):
    names = [str(pair[0]) for pair in attributes]
    if xml_path is not None:
        labels = _xml_label_names(xml_path)
        indices = []
        for label in labels:
            if label not in names:
                raise ValueError(f"XML label {label!r} missing in ARFF attributes")
            indices.append(names.index(label))
        return indices
    # MEKA uses -C q for FIRST q labels, and -C -q for LAST q labels.
    m = re.search(r"-C\s+([+-]?\d+)", relation, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Label definition absent: pass --xml for Mulan, or use MEKA -C q in @relation")
    q = int(m.group(1))
    if not q or abs(q) >= len(names):
        raise ValueError(f"Invalid MEKA label count {q} for {len(names)} attributes")
    return list(range(q)) if q > 0 else list(range(len(names) + q, len(names)))


def read_benchmark_arff(arff_path: str | Path, dataset: str, xml_path: str | Path | None = None) -> BenchmarkData:
    try:
        import arff  # liac-arff, handles additional ARFF variants
    except ImportError:
        arff = None

    path = Path(arff_path)
    if not path.is_file():
        raise FileNotFoundError(f"ARFF file not found: {path}")
    xml = Path(xml_path) if xml_path is not None else None
    if xml is not None and not xml.is_file():
        raise FileNotFoundError(f"XML file not found: {xml}")
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if arff is None:
        obj = _read_arff_stdlib(path)
    else:
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            obj = arff.load(stream)
    attributes = obj["attributes"]
    col_names = [name for name, _ in attributes]
    label_indices = _extract_label_indices(str(obj.get("relation", "")), attributes, xml)
    label_set = set(label_indices)
    feat_indices = [i for i in range(len(attributes)) if i not in label_set]
    if not feat_indices:
        raise ValueError("No feature attributes")

    # One-hot expand nominal features using ONLY schema-defined categories.
    feat_kinds = []
    encoded_features = 0
    for i in feat_indices:
        attr_type = attributes[i][1]
        if isinstance(attr_type, (list, tuple)):
            cats = [str(c) for c in attr_type]
            if len(set(cats)) != len(cats):
                raise ValueError(f"Duplicate nominal categories for {col_names[i]}")
            feat_kinds.append(("nominal", cats))
            encoded_features += len(cats)
        elif str(attr_type).upper() in ("NUMERIC", "REAL", "INTEGER"):
            feat_kinds.append(("numeric", None))
            encoded_features += 1
        else:
            raise ValueError(f"Unsupported feature type: {col_names[i]} / {attr_type}; encode manually")

    records = []
    for row_number, row in enumerate(obj["data"], start=1):
        if len(row) != len(attributes):
            raise ValueError(f"Row {row_number} has {len(row)} values; expected {len(attributes)}")
        y = []
        for i in label_indices:
            val = row[i]
            if val is None or str(val).strip() not in ("0", "1", "0.0", "1.0"):
                raise ValueError(f"Invalid or missing binary label at row {row_number}, column {col_names[i]}: {val}")
            y.append(int(float(val)))
        x = []
        for i, (kind, categories) in zip(feat_indices, feat_kinds):
            val = row[i]
            if val is None or val == "?":
                raise ValueError(f"Missing feature at row {row_number}, column {col_names[i]}. Do not silently impute.")
            if kind == "nominal":
                value = str(val)
                if value not in categories:
                    raise ValueError(f"Unknown category {value} at row {row_number}, {col_names[i]}")
                x.extend(float(value == c) for c in categories)
            else:
                number = float(val)
                if not math.isfinite(number):
                    raise ValueError(f"Non-finite value at row {row_number}, {col_names[i]}")
                x.append(number)
        records.append((tuple(x), tuple(y)))
    if not records:
        raise ValueError("No data rows")
    return BenchmarkData(
        name=dataset, records=records, raw_features=len(feat_indices),
        encoded_features=encoded_features, n_labels=len(label_indices),
        label_names=tuple(col_names[i] for i in label_indices),
        arff_sha256=source_hash,
        encoding="numeric unchanged; nominal one-hot by ARFF schema; no missing-value imputation",
    )


def verify_paper_shape(data: BenchmarkData) -> tuple[bool, list[str]]:
    info = PAPER_DATASETS.get(data.name.lower())
    if info is None:
        return False, [f"No per-dataset paper reference registered for {data.name}"]
    mismatches = []
    for key, actual in (("n", len(data.records)), ("features", data.raw_features), ("labels", data.n_labels)):
        if actual != info[key]:
            mismatches.append(f"{key}: paper={info[key]}, input={actual}")
    return not mismatches, mismatches
