from __future__ import annotations

import argparse
import json

from src.explain import DEFAULT_TOP_K, explain_prediction
from src.final_artifact import load_final_artifact_components, predict_final_artifact


def load_unified_artifact_components(artifact_dir=None):
    return load_final_artifact_components() if artifact_dir is None else load_final_artifact_components(artifact_dir)


def predict_unified(subject: str, body: str, components=None, *, explanation_top_k: int = DEFAULT_TOP_K):
    loaded = components or load_final_artifact_components()
    result = predict_final_artifact(subject, body, loaded)
    result["text_explanation"] = explain_prediction(
        subject,
        body,
        top_k=explanation_top_k,
        explanation_artifacts=loaded,
        expected_result=result,
    )
    return result


__all__ = [
    "load_final_artifact_components",
    "predict_final_artifact",
    "load_unified_artifact_components",
    "predict_unified",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step-25C final text-led phishing artifact")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--explanation-top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)
    print(json.dumps(predict_unified(
        args.subject, args.body, explanation_top_k=args.explanation_top_k
    ), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
