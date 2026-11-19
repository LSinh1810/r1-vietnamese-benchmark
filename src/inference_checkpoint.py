"""Checkpoint JSONL theo batch và tệp JSONL tổng hợp sau khi hoàn thành."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def _atomic_write(path, lines):
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_manifest(path, manifest):
    """Gọi dưới khóa của lần chạy để ngăn hai phiên cùng ghi."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cấu hình hoặc dataset đã đổi. Hãy dùng run_id mới.")
    else:
        _atomic_write(path, [json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"])


def batch_is_complete(path, expected_keys):
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as stream:
        keys = [(row["question_id"], row["generation_index"])
                for row in map(json.loads, stream)]
    if keys != expected_keys:
        raise ValueError(f"Checkpoint không khớp với batch dự kiến: {path.name}")
    return True


def save_batch(path, rows):
    if path.exists():
        raise FileExistsError(path)
    _atomic_write(path, (json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _batch_lines(batch_paths):
    for path in batch_paths:
        with path.open(encoding="utf-8") as stream:
            yield from stream


def finalize_results(path, batch_paths):
    if not path.exists():
        _atomic_write(path, _batch_lines(batch_paths))
