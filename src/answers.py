"""Trích xuất đáp án và so sánh số hoặc biểu thức bằng số hữu tỉ chính xác."""

import ast
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ExtractedAnswer:
    value: str | None
    status: str  # extracted, missing, hoặc review


ANSWER_LABEL = re.compile(
    r"^[ \t]*\*{0,2}(?:đáp án(?: cuối cùng)?|final answer|answer)"
    r"\*{0,2}[ \t]*[:：]\*{0,2}[ \t]*", re.I | re.M,
)


def split_response(response, prompt=""):
    """Trả về (suy luận quan sát được hoặc None, văn bản trả lời cuối)."""
    has_open = prompt.rstrip().endswith("<think>") or response.lstrip().startswith("<think>")
    body = response.lstrip().removeprefix("<think>")
    if has_open and "</think>" in body:
        reasoning, final = body.split("</think>", 1)
        return reasoning.strip(), final.strip()
    if has_open:
        return None, ""
    return None, response.strip()


def _boxed_value(text):
    start = len(r"\boxed{")
    depth = 1
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip(), text[index + 1:]
    return "", text


def extract_answer(response, prompt="", *, truncated=False):
    """Ưu tiên nhãn cuối hoặc boxed rõ ràng. Ca mơ hồ được đưa vào hàng đợi."""
    _, final = split_response(response, prompt)
    if truncated or (not final and response.strip() and "</think>" not in response):
        return ExtractedAnswer(None, "review")
    matches = list(ANSWER_LABEL.finditer(final))
    if matches:
        last = matches[-1]
        answer_line, _, remainder = final[last.end():].partition("\n")
        candidate = answer_line.strip(" *\r")
        if len(matches) != 1 or remainder.strip():
            return ExtractedAnswer(candidate or None, "review")
    else:
        start = final.find(r"\boxed{")
        if start > 0 and final[:start].rsplit("\n", 1)[-1].strip(" $"):
            return ExtractedAnswer(None, "review")
        candidate = final[start:] if start >= 0 else ""

    candidate = candidate.removeprefix("$").removesuffix("$").strip()
    if candidate.startswith(r"\boxed{"):
        candidate, tail = _boxed_value(candidate)
        if tail.strip(" .$\t") or not candidate:
            return ExtractedAnswer(candidate or None, "review")
    if candidate and not re.search(r"\b(?:hoặc|or)\b|[;?]", candidate, re.I):
        return ExtractedAnswer(candidate, "extracted")
    if not final or (matches and not candidate):
        return ExtractedAnswer(None, "missing")
    return ExtractedAnswer(candidate or None, "review")


def _text(value):
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def _math_source(value):
    text = _text(value).strip("$ ").rstrip(".")
    text = text.removeprefix(r"\(").removesuffix(r"\)")
    for old, new in {"−": "-", "×": "*", "÷": "/", r"\times": "*", r"\cdot": "*", "^": "**"}.items():
        text = text.replace(old, new)
    fraction = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    for _ in range(8):
        updated = fraction.sub(r"((\1)/(\2))", text)
        if updated == text:
            break
        text = updated
    if text.endswith(("%", r"\%")):
        percent_value = text.removesuffix("%").removesuffix("\\")
        text = f"({percent_value})/100"
    return re.sub(r"(?<=\d),(?=\d)", ".", text).strip()


def _evaluate_node(node, source):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        literal = ast.get_source_segment(source, node)
        if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]{1,2})?", literal):
            raise ValueError("Dạng số ngoài phạm vi.")
        return Fraction(literal)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, source)
        return -value if isinstance(node.op, ast.USub) else value
    if not isinstance(node, ast.BinOp):
        raise ValueError("Biểu thức ngoài phạm vi.")
    left = _evaluate_node(node.left, source)
    right = _evaluate_node(node.right, source)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if (isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 12
            and max(left.numerator.bit_length(), left.denominator.bit_length()) <= 512):
        return left ** int(right)
    raise ValueError("Toán tử hoặc số mũ ngoài phạm vi.")


def _number(value):
    source = _math_source(value)
    if len(source) > 256 or not re.fullmatch(r"[0-9.e+*/()\-\s]+", source):
        return None
    try:
        if re.fullmatch(r"[+-]?0[0-9]+", source):
            return Fraction(source)
        return _evaluate_node(ast.parse(source, mode="eval").body, source)
    except (SyntaxError, ValueError, ArithmeticError, RecursionError):
        return None


def compare_answers(candidate, gold, *, accepted_answers=(), absolute_tolerance=None):
    """Trả True, False hoặc None nếu cần chấm thủ công. Dung sai mặc định bằng 0."""
    tolerance = Fraction(0) if absolute_tolerance is None else _number(absolute_tolerance)
    if tolerance is None or tolerance < 0:
        raise ValueError("absolute_tolerance phải là một số không âm.")
    actual = _number(candidate)
    uncertain = False
    for expected in (gold, *accepted_answers):
        reference = _number(expected)
        if actual is not None and reference is not None:
            if abs(actual - reference) <= tolerance:
                return True
            continue
        if _text(candidate).rstrip(".") == _text(expected).rstrip("."):
            return True
        # Đơn vị, biểu thức ký hiệu và định dạng số ngoài phạm vi cần người chấm.
        if any(re.search(r"\d|[=\\^√]", str(value)) for value in (candidate, expected)):
            uncertain = True
    return None if uncertain else False
