import unittest

from src.answers import compare_answers, extract_answer, split_response
from src.evaluation import (
    accuracy_metrics, error_analysis, identify_language, language_analysis,
    length_accuracy_relationship, reasoning_length_metrics, score_predictions,
)


class AnswerTests(unittest.TestCase):
    def test_numeric_equivalence(self):
        for value in ("0,5", r"\frac{2}{4}", "50%", "(3 - 2)^2 ÷ 2", "5e-1"):
            with self.subTest(value=value):
                self.assertIs(compare_answers(value, "1/2"), True)
        self.assertIs(compare_answers("0.33", "1/3"), False)
        self.assertIs(compare_answers("0.33", "1/3", absolute_tolerance="0.01"), True)
        self.assertIs(compare_answers(" A ", "B", accepted_answers=["a"]), True)

    def test_unsupported_math_is_reviewed(self):
        self.assertIsNone(compare_answers("2 m", "200 cm"))
        self.assertIsNone(compare_answers("sqrt(4)", "2"))
        self.assertIsNone(compare_answers("1/0", "2"))

    def test_split_and_extract(self):
        response = "Tính 2/4. </think>\n**Đáp án:** \\boxed{\\frac{2}{4}}"
        self.assertEqual(split_response(response, "prompt<think>\n")[0], "Tính 2/4.")
        result = extract_answer(response, "prompt<think>\n")
        self.assertEqual((result.status, result.value), ("extracted", r"\frac{2}{4}"))

    def test_ambiguous_or_missing_answers(self):
        for response in ("Có thể là 4", "Đáp án: 2\nĐáp án: 4", "Đáp án: 2 hoặc 4",
                         "Đáp án:\n      4", r"\boxed{\frac{2}{4}", r"Không phải \boxed{4}"):
            with self.subTest(response=response):
                self.assertEqual(extract_answer(response).status, "review")
        self.assertEqual(extract_answer("Đáp án: 4", truncated=True).status, "review")
        self.assertEqual(extract_answer("Tính 2/4.</think>", "<think>\n").status, "missing")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.questions = {"001": {"question": "Unit test", "answer": "1/2", "topic": "test", "difficulty": "dễ"}}
        base = {"question_id": "001", "prompt": "", "stop_reason": "eos"}
        self.predictions = [
            dict(base, generation_index=1, generated_response="Đáp án: 50%", generated_token_count=5),
            dict(base, generation_index=2, generated_response="Có thể là 2.", generated_token_count=9),
        ]

    def test_accuracy_with_pending_and_missing(self):
        rows = score_predictions(self.predictions, self.questions, 3)
        stats = accuracy_metrics(rows, 3)
        self.assertIsNone(stats["accuracy"])
        self.assertEqual(stats["observed_accuracy_bounds"], (0.5, 1.0))
        self.assertEqual((stats["pending_review"], stats["missing_inference"]), (1, 1))

    def test_manual_review_and_analysis(self):
        review = {"question_id": "001", "generation_index": 2, "correct": False,
                  "error_tags": ["unsupported_conclusion", "answer_not_extracted"]}
        rows = score_predictions(self.predictions, self.questions, 2, reviews=[review])
        self.assertEqual(accuracy_metrics(rows, 2)["accuracy"], 0.5)
        self.assertEqual(reasoning_length_metrics(rows)["generated_token_count"]["mean"], 7)
        relation = length_accuracy_relationship(rows, length_field="generated_token_count")
        self.assertAlmostEqual(relation["point_biserial_r"], -1.0)
        self.assertEqual(error_analysis(rows)["manual_share_reviewed_wrong"]["unsupported_conclusion"], 1)

    def test_reject_duplicate_result(self):
        with self.assertRaisesRegex(ValueError, "trùng"):
            score_predictions([self.predictions[0]] * 2, self.questions, 2)

    def test_language_and_denominators(self):
        self.assertEqual(identify_language("Kết quả là bốn"), "vi")
        self.assertEqual(identify_language("Therefore the result is four"), "en")
        self.assertEqual(identify_language("Kết quả là bốn. Therefore the result is four"), "mixed_vi_en")
        self.assertEqual(identify_language("42"), "unknown")
        prediction = dict(self.predictions[0], prompt="<think>", generated_response=(
            "Kết quả là bốn.</think>\nĐáp án: Therefore the result is four"))
        rows = score_predictions([prediction], self.questions, 1, count_tokens=lambda text: len(text.split()))
        self.assertEqual(rows[0]["reasoning_token_count"], 4)
        self.assertEqual(rows[0]["final_language"], "en")
        self.assertEqual(language_analysis(rows)["mixed_anywhere_share_all"], 1)
