"""Chấm đầu ra inference và tính các thống kê theo giao thức nghiên cứu."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import StatisticsError, correlation, mean

from .answers import ANSWER_LABEL, compare_answers, extract_answer, split_response


ERROR_LABELS = (
    "knowledge", "calculation", "misunderstood_question", "inconsistent_reasoning",
    "unsupported_conclusion", "answer_not_extracted",
)
LANGUAGE_LABELS = ("vi", "en", "mixed_vi_en", "unknown")
_VI_WORDS = frozenset("và của với nếu thì đáp án kết quả không chúng ta nên được có từ này là vậy suy ra bài toán".split())
_EN_WORDS = frozenset("the and from with therefore because this these those then would should could answer result calculate we must since".split())
_VI_LETTERS = frozenset("đăâêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


def identify_language(text):
    """Heuristic từ chỉ dấu, ưu tiên unknown khi đoạn quá ngắn hoặc thiếu tín hiệu."""
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = re.sub(r"\$[^$]*\$|\\[a-z]+|</?[a-z]+>", " ", text)
    words = re.findall(r"[^\W\d_]+", text)
    vi = {word for word in words if word in _VI_WORDS or _VI_LETTERS.intersection(word)}
    en = set(words) & _EN_WORDS
    if len(words) < 3:
        return "unknown"
    if len(vi) >= 2 and len(en) >= 2:
        return "mixed_vi_en"
    if len(vi) >= 2:
        return "vi"
    if len(en) >= 2:
        return "en"
    return "unknown"


def language_analysis(rows):
    """Tính phân phối riêng cho suy luận và trả lời cuối, với hai mẫu số."""
    result = {}
    for field in ("reasoning_language", "final_language"):
        counts = Counter(row[field] for row in rows)
        total = sum(counts.values())
        identified = total - counts["unknown"]
        result[field] = {
            "count": {label: counts[label] for label in LANGUAGE_LABELS},
            "share_all": {label: counts[label] / total if total else None for label in LANGUAGE_LABELS},
            "share_identified": {
                label: counts[label] / identified if identified else None
                for label in LANGUAGE_LABELS if label != "unknown"
            },
        }
    combined = sum(
        "mixed_vi_en" in (row["reasoning_language"], row["final_language"])
        or {row["reasoning_language"], row["final_language"]} == {"vi", "en"}
        for row in rows
    )
    result["mixed_anywhere_share_all"] = combined / len(rows) if rows else None
    return result


def accuracy_metrics(rows, expected_count):
    """Một lần sinh mỗi quan sát. Điểm chính chỉ có khi đủ các lượt đã định."""
    observed = len(rows)
    correct = sum(row["correct"] is True for row in rows)
    pending = sum(row["correct"] is None for row in rows)
    missing = expected_count - observed
    if missing < 0:
        raise ValueError("Số đầu ra vượt thiết kế.")
    return {
        "expected": expected_count, "observed": observed, "correct": correct,
        "wrong": observed - correct - pending, "pending_review": pending,
        "missing_inference": missing,
        "accuracy": correct / observed if observed and not (pending or missing) else None,
        "observed_accuracy_bounds": (
            (correct / observed, (correct + pending) / observed) if observed else None
        ),
    }


def reasoning_length_metrics(rows):
    """Tóm tắt token đã ghi và token suy luận được mã hóa lại nếu có tokenizer."""
    summary = {}
    for field in ("generated_token_count", "reasoning_token_count", "response_char_count"):
        values = [row[field] for row in rows if row[field] is not None]
        summary[field] = {"n": len(values), "mean": mean(values) if values else None}
    return summary


def length_accuracy_relationship(rows, *, length_field="reasoning_token_count", exclude_truncated=False):
    """Tương quan point-biserial mô tả, chỉ dùng lượt có độ dài và điểm xác định."""
    pairs = [
        (row[length_field], int(row["correct"])) for row in rows
        if row[length_field] is not None and row["correct"] is not None
        and not (exclude_truncated and row["stop_reason"] == "max_new_tokens")
    ]
    correct_lengths = [length for length, score in pairs if score]
    wrong_lengths = [length for length, score in pairs if not score]
    coefficient = None
    if correct_lengths and wrong_lengths:
        try:
            lengths, scores = zip(*pairs)
            coefficient = correlation(lengths, scores)
        except StatisticsError:
            pass
    return {
        "n": len(pairs), "point_biserial_r": coefficient,
        "mean_correct": mean(correct_lengths) if correct_lengths else None,
        "mean_wrong": mean(wrong_lengths) if wrong_lengths else None,
    }


def error_analysis(rows):
    """Tổng hợp nhãn thủ công và sự kiện trích xuất tự động, với mẫu số rõ ràng."""
    reviewed = [row for row in rows if row["error_tags"] is not None]
    counts = Counter(tag for row in reviewed for tag in row["error_tags"])
    wrong = [row for row in rows if row["correct"] is False]
    reviewed_wrong = [row for row in reviewed if row["correct"] is False]
    wrong_counts = Counter(tag for row in reviewed_wrong for tag in row["error_tags"])
    not_extracted = sum(row["extraction_status"] != "extracted" for row in rows)
    return {
        "reviewed_count": len(reviewed), "wrong_count": len(wrong),
        "reviewed_wrong_count": len(reviewed_wrong), "manual_counts": dict(counts),
        "manual_share_reviewed": {label: counts[label] / len(reviewed) if reviewed else None
                                  for label in ERROR_LABELS},
        "manual_share_reviewed_wrong": {
            label: wrong_counts[label] / len(reviewed_wrong) if reviewed_wrong else None
            for label in ERROR_LABELS
        },
        "automatic_extraction_issue_count": not_extracted,
        "automatic_extraction_issue_share_all": not_extracted / len(rows) if rows else None,
    }


def _index_by_key(records):
    indexed = {}
    for record in records:
        question_id = record["question_id"]
        generation = record["generation_index"]
        if type(question_id) not in (str, int) or type(generation) is not int or generation < 1:
            raise ValueError("question_id hoặc generation_index không hợp lệ.")
        key = (question_id, generation)
        if key in indexed:
            raise ValueError(f"Bản ghi trùng: {key!r}")
        indexed[key] = record
    return indexed


def score_predictions(predictions, questions, expected_repeats, *, reviews=(), count_tokens=None):
    """Chấm các lượt quan sát được. reviews chứa quyết định độc lập của người chấm."""
    indexed = _index_by_key(predictions)
    manual = _index_by_key(reviews)
    if manual.keys() - indexed.keys():
        raise ValueError("Review có khóa không thuộc đầu ra đã quan sát.")

    rows = []
    for (question_id, generation_index), prediction in indexed.items():
        if question_id not in questions or generation_index > expected_repeats:
            raise ValueError(f"Đầu ra ngoài thiết kế: {(question_id, generation_index)!r}")
        item = questions[question_id]
        response = prediction["generated_response"]
        prompt = prediction["prompt"]
        reasoning, final = split_response(response, prompt)
        extracted = extract_answer(response, prompt, truncated=prediction["stop_reason"] == "max_new_tokens")
        correct = None
        if extracted.status == "missing":
            correct = False
        elif extracted.status == "extracted":
            correct = compare_answers(extracted.value, item["answer"],
                                      accepted_answers=item.get("accepted_answers", ()),
                                      absolute_tolerance=item.get("absolute_tolerance"))
        review = manual.get((question_id, generation_index), {})
        if "correct" in review:
            if type(review["correct"]) is not bool:
                raise ValueError("correct trong review phải là true hoặc false.")
            correct = review["correct"]
        error_tags = review.get("error_tags")
        if error_tags is not None and (not isinstance(error_tags, list)
                or any(tag not in ERROR_LABELS for tag in error_tags)
                or len(set(error_tags)) != len(error_tags)):
            raise ValueError("error_tags phải là danh sách nhãn hợp lệ, không trùng.")
        rows.append({
            **prediction, "correct": correct, "needs_manual_review": correct is None,
            "extraction_status": extracted.status, "extracted_answer": extracted.value,
            "gold_answer": item["answer"], "question": item["question"],
            "topic": item["topic"], "difficulty": item["difficulty"],
            "reasoning_token_count": count_tokens(reasoning) if reasoning and count_tokens else None,
            "response_char_count": len(response), "reasoning_language": identify_language(reasoning),
            "final_language": identify_language(ANSWER_LABEL.sub("", final)), "error_tags": error_tags,
        })
    return rows


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_run_predictions(run_dir):
    """Đọc dự đoán tổng hợp hoặc các batch đã lưu của một lần chạy."""
    run_dir = Path(run_dir)
    final_path = run_dir / "predictions.jsonl"
    paths = [final_path] if final_path.is_file() else sorted((run_dir / "batches").glob("*.jsonl"))
    return [row for path in paths for row in read_jsonl(path)]


def load_evaluation_inputs(run_dir, dataset_path):
    """Liên kết predictions với dataset gốc bằng SHA-256 và manifest."""
    from .dataset import load_dataset, normalize_dataset, validate_dataset

    run_dir = Path(run_dir)
    dataset_path = Path(dataset_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset không khớp manifest.")
    data = normalize_dataset(load_dataset(dataset_path))
    if not validate_dataset(data).empty or len(data) != manifest["question_count"]:
        raise ValueError("Dataset có lỗi hoặc sai số lượng so với manifest.")
    questions = {}
    for item in data.to_dict("records"):
        alternatives = item.get("accepted_answers")
        if isinstance(alternatives, str) and alternatives.strip():
            alternatives = json.loads(alternatives)
        if isinstance(alternatives, list):
            item["accepted_answers"] = alternatives
        elif alternatives is None or alternatives == "" or alternatives != alternatives:
            item["accepted_answers"] = []
        else:
            raise ValueError("accepted_answers phải là mảng JSON.")
        tolerance = item.get("absolute_tolerance")
        if tolerance == "" or tolerance != tolerance:
            item["absolute_tolerance"] = None
        questions[item["id"]] = item

    predictions = read_run_predictions(run_dir)
    if any(row["model_name"] != manifest["model"]["id"] for row in predictions):
        raise ValueError("Model trong dự đoán không khớp manifest.")
    return manifest, questions, predictions
