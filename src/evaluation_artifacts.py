"""Lưu và đọc điểm đã chấm để tái sử dụng trong notebook trực quan hóa."""

import hashlib
import json
from pathlib import Path

from .evaluation import _index_by_key, read_run_predictions


_SCORED_FIELDS = (
    "question_id", "generation_index", "topic", "difficulty", "correct",
    "reasoning_token_count", "reasoning_language", "final_language", "error_tags",
    "extraction_status", "response_char_count",
)


def _prediction_digest(predictions):
    lines = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in predictions)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def save_evaluation(path, manifest, questions, predictions, scored_rows):
    """Lưu điểm từng lượt và thiết kế mẫu, kể cả những câu chưa có đầu ra."""
    artifact = {
        "format_version": 1,
        "manifest": manifest,
        "prediction_sha256": _prediction_digest(predictions),
        "questions": [{"question_id": key, "topic": item["topic"], "difficulty": item["difficulty"]}
                      for key, item in questions.items()],
        "scores": [{field: row[field] for field in _SCORED_FIELDS} for row in scored_rows],
    }
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) == artifact:
            return path
        raise FileExistsError(f"Đã có kết quả khác tại {path}. Chọn tên tệp đánh giá mới.")
    payload = json.dumps(artifact, ensure_ascii=False, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
    return path


def load_saved_evaluation(run_dir, evaluation_path=None):
    """Ghép điểm với đầu ra inference, đồng thời kiểm tra hai bản lưu cùng lần chạy."""
    run_dir = Path(run_dir)
    path = Path(evaluation_path) if evaluation_path else run_dir / "evaluation.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if artifact["format_version"] != 1 or artifact["manifest"] != manifest:
        raise ValueError(f"Tệp evaluation không tương thích với manifest của {run_dir.name}.")

    predictions = read_run_predictions(run_dir)
    if _prediction_digest(predictions) != artifact["prediction_sha256"]:
        raise ValueError(f"Đầu ra của {run_dir.name} đã đổi. Lưu một bản evaluation mới từ notebook 03.")

    inferred = _index_by_key(predictions)
    scores = _index_by_key(artifact["scores"])
    questions = {row["question_id"]: row for row in artifact["questions"]}
    repeats = len(manifest["inference"]["seeds"])
    if (inferred.keys() != scores.keys() or len(questions) != manifest["question_count"]
            or any(question_id not in questions or index > repeats for question_id, index in scores)):
        raise ValueError(f"Điểm hoặc danh sách câu hỏi của {run_dir.name} không khớp thiết kế.")
    rows = [dict(inferred[key], **score) for key, score in scores.items()]
    return manifest, list(questions.values()), rows
