"""Đọc, chuẩn hóa và kiểm tra dữ liệu câu hỏi suy luận."""

import csv
import json
import math
import unicodedata
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = ("id", "question", "answer", "topic", "difficulty")
DIFFICULTIES = ("dễ", "trung bình", "khó")
GROUP_FIELDS = ("topic", "difficulty")
ISSUE_COLUMNS = ("row", "field", "code", "detail")


def _read_csv(stream):
    reader = csv.reader(stream, strict=True)
    header = next(reader, [])
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV phải có tiêu đề và tên cột không được để trống.")
    if len(set(header)) != len(header):
        raise ValueError("CSV có tên cột trùng lặp.")

    records = []
    for values in reader:
        if not values:
            continue
        if len(values) != len(header):
            raise ValueError(f"CSV, dòng {reader.line_num}: sai số lượng trường.")
        records.append(dict(zip(header, values)))
    return pd.DataFrame(records, columns=header, dtype=object)


def _unique_json_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"khóa trùng lặp {key!r}")
        record[key] = value
    return record


def _reject_json_constant(value):
    raise ValueError(f"hằng số JSON không hợp lệ {value}")


def load_dataset(path):
    """Đọc UTF-8 (có thể có BOM). CSV giữ nguyên chuỗi, kể cả ID có số 0 đầu."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise ValueError(f"Định dạng không được hỗ trợ: {path.suffix!r}")

    with path.open(encoding="utf-8-sig", newline="") as stream:
        if suffix == ".csv":
            try:
                return _read_csv(stream)
            except csv.Error as error:
                raise ValueError(f"CSV không hợp lệ: {error}") from error

        records = []
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except ValueError as error:
                raise ValueError(f"JSONL, dòng {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL, dòng {line_number}: cần một đối tượng JSON.")
            records.append(record)
    return pd.DataFrame(records, dtype=object)


def _normalize_label(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def normalize_dataset(dataset):
    """Trả về bản sao. Giữ nguyên nội dung, dấu câu và xuống dòng bên trong đề."""
    normalized = dataset.copy()
    for field in REQUIRED_FIELDS:
        if field not in normalized:
            continue
        normalize = _normalize_label if field in GROUP_FIELDS else str.strip
        # Giữ kiểu gốc của ID số nguyên ngay cả khi cột có giá trị thiếu.
        normalized[field] = pd.Series(
            [normalize(value) if isinstance(value, str) else value for value in dataset[field]],
            index=dataset.index,
            dtype=object,
        )
    return normalized


def _is_missing(value):
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, float) and math.isnan(value))
    )


def _is_valid_value(field, value):
    if isinstance(value, str):
        return field != "difficulty" or value in DIFFICULTIES
    if type(value) is int:
        return field in {"id", "answer"}
    return field == "answer" and type(value) is float and math.isfinite(value)


def validate_dataset(dataset, *, allowed_topics=None):
    """Trả bảng lỗi, giữ tất cả bản ghi. row là thứ tự bản ghi từ 1."""
    normalized = normalize_dataset(dataset)
    topics = None if allowed_topics is None else {_normalize_label(t) for t in allowed_topics}
    issues = []
    if len(normalized.index) == 0:
        issues.append((None, None, "empty_dataset", "Dataset không có bản ghi."))

    seen = {"id": {}, "question": {}}
    for field in REQUIRED_FIELDS:
        if field not in normalized:
            issues.append((None, field, "missing_column", "Thiếu cột bắt buộc."))
            continue

        for row, value in enumerate(normalized[field], start=1):
            if _is_missing(value):
                issues.append((row, field, "missing_value", "Thiếu giá trị bắt buộc."))
            elif not _is_valid_value(field, value):
                detail = "Sai kiểu hoặc giá trị không hữu hạn."
                if field == "difficulty":
                    detail = f"Độ khó hợp lệ: {', '.join(DIFFICULTIES)}."
                issues.append((row, field, "invalid_value", detail))
            elif field == "topic" and topics is not None and value not in topics:
                issues.append((row, field, "unknown_topic", "Nhãn ngoài danh mục chủ đề."))
            elif field in seen:
                key = _normalize_label(value) if field == "question" else str(value)
                first_row = seen[field].setdefault(key, row)
                if first_row != row:
                    issues.append((row, field, f"duplicate_{field}", f"Trùng với bản ghi {first_row}."))

    return pd.DataFrame(issues, columns=ISSUE_COLUMNS).astype({"row": "Int64"})


def group_dataset(dataset, *fields):
    """Nhóm theo topic, difficulty hoặc cả hai. Giữ nhóm giá trị thiếu."""
    if not fields or len(set(fields)) != len(fields) or any(f not in GROUP_FIELDS for f in fields):
        raise ValueError("Chọn topic, difficulty hoặc cả hai trường khác nhau.")
    normalized = normalize_dataset(dataset)
    for field in fields:
        values = normalized.get(field, pd.Series(index=normalized.index, dtype=object))
        normalized[field] = values.map(
            lambda value: value if isinstance(value, str) and value else pd.NA
        )
    keys = fields[0] if len(fields) == 1 else list(fields)
    return normalized.groupby(keys, dropna=False, sort=True)


def dataset_distribution(dataset, *fields):
    """Đếm bản ghi theo nhóm, bao gồm các bản ghi cần rà soát."""
    return group_dataset(dataset, *fields).size().rename("count").reset_index()
