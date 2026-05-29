from typing import Dict, List


def evaluate_answer(answer: str, expected: str) -> Dict[str, object]:
    normalized_answer = answer.strip()
    normalized_expected = expected.strip()
    return {
        "exact_match": normalized_answer == normalized_expected,
        "answer_length": len(normalized_answer),
        "expected_length": len(normalized_expected),
    }


def summarize_sources(sources: List[Dict]) -> Dict[str, int]:
    return {
        "source_count": len(sources),
        "top_source_pages": [source.get("page") for source in sources[:3]],
    }
