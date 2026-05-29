"""
evaluation/evaluator.py

Production evaluation using lightweight metrics.
Full RAGAS integration available when ragas package is installed.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("mentor-x-ai.evaluation")


def evaluate_answer(answer: str, expected: str) -> Dict[str, object]:
    """Basic evaluation metrics."""
    normalized_answer = answer.strip().lower()
    normalized_expected = expected.strip().lower()

    # Token overlap (simple F1)
    answer_tokens = set(normalized_answer.split())
    expected_tokens = set(normalized_expected.split())

    if not expected_tokens:
        token_f1 = 0.0
    else:
        intersection = answer_tokens & expected_tokens
        precision = (
            len(intersection) / len(answer_tokens)
            if answer_tokens
            else 0.0
        )
        recall = len(intersection) / len(expected_tokens)

        token_f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return {
        "exact_match": normalized_answer == normalized_expected,
        "token_f1": round(token_f1, 4),
        "answer_length": len(answer.strip()),
        "expected_length": len(expected.strip()),
    }


def evaluate_faithfulness(answer: str, context: str) -> Dict[str, object]:
    """
    Lightweight faithfulness check.
    Checks what fraction of answer sentences are grounded in context.
    """
    if not context or not answer:
        return {
            "faithfulness_score": 0.0,
            "grounded_sentences": 0,
            "total_sentences": 0,
        }

    sentences = [
        s.strip()
        for s in answer.split(".")
        if len(s.strip()) > 10
    ]

    if not sentences:
        return {
            "faithfulness_score": 1.0,
            "grounded_sentences": 0,
            "total_sentences": 0,
        }

    context_lower = context.lower()
    grounded = 0

    for sentence in sentences:
        words = sentence.lower().split()

        if len(words) < 3:
            grounded += 1
            continue

        ngrams = [
            " ".join(words[i:i + 3])
            for i in range(len(words) - 2)
        ]

        if any(ngram in context_lower for ngram in ngrams):
            grounded += 1

    score = grounded / len(sentences)

    return {
        "faithfulness_score": round(score, 4),
        "grounded_sentences": grounded,
        "total_sentences": len(sentences),
    }


def evaluate_rag_response(
    question: str,
    answer: str,
    context: str,
    sources: List[Dict],
    expected: Optional[str] = None,
) -> Dict[str, object]:
    """
    Full evaluation of a RAG response.
    Call this after every retriever.ask() in testing/evaluation mode.
    """

    result: Dict[str, object] = {
        "question_length": len(question),
        "answer_length": len(answer),
        "sources_count": len(sources),
        "has_fallback": "couldn't find" in answer.lower(),
        **evaluate_faithfulness(answer, context),
    }

    if expected:
        result.update(evaluate_answer(answer, expected))

    if sources:
        scores = [s.get("score", 0) for s in sources]

        result["avg_source_score"] = round(
            sum(scores) / len(scores),
            4,
        )

        result["top_source_score"] = round(
            max(scores),
            4,
        )

    logger.info(
        "RAG eval: faithfulness=%.2f sources=%d fallback=%s",
        result.get("faithfulness_score", 0),
        result["sources_count"],
        result["has_fallback"],
    )

    return result


def summarize_sources(sources: List[Dict]) -> Dict[str, object]:
    return {
        "source_count": len(sources),
        "top_source_pages": [
            s.get("page")
            for s in sources[:3]
        ],
        "avg_score": (
            round(
                sum(s.get("score", 0) for s in sources)
                / len(sources),
                4,
            )
            if sources
            else 0
        ),
    }
