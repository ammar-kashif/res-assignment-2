"""Three-layer moderation pipeline: regex pre-filter -> calibrated DistilBERT
-> human-review queue.

Usage
-----
    from pipeline import ModerationPipeline

    pipe = ModerationPipeline.from_pretrained(
        checkpoint_dir="checkpoints/baseline",
        isotonic_path="checkpoints/baseline/isotonic.pkl",
    )
    print(pipe.predict("you should kill yourself"))
    # {"decision": "block", "confidence": 1.0, "triggered_layer": 1, "category": "self_harm"}

The calibrator is a `sklearn.isotonic.IsotonicRegression` fit on raw DistilBERT
softmax probabilities; persist it with `joblib.dump` after training.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

# ---------------------------------------------------------------------------
# Layer 1: regex pre-filter
# ---------------------------------------------------------------------------
BLOCKLIST: dict[str, list[re.Pattern]] = {
    "threats": [
        re.compile(r"\bi\s*(?:'ll|will|am gonna|am going to|gonna)\s+(kill|murder|shoot|stab|hurt)\s+you\b", re.I),
        re.compile(r"\byou(?:'re| are)\s+going to die\b", re.I),
        re.compile(r"\b(?:someone should|i hope someone)\s+(?:kill|shoot|hurt|stab)s?\s+(?:you|him|her|them)\b", re.I),
        re.compile(r"\bi('ll| will)\s+find (?:where you live|out where you are)\b", re.I),
        re.compile(r"\byou (?:deserve to|should) (?:be )?(?:beaten|hurt|killed)\b", re.I),
    ],
    "self_harm": [
        re.compile(r"\b(?:go\s+)?kill yourself\b", re.I),
        re.compile(r"\byou should (?:kill|hurt|harm) yourself\b", re.I),
        re.compile(r"\bnobody would miss you\b", re.I),
        re.compile(r"\bdo (?:everyone|us all) a favou?r and (?:disappear|die|leave)\b", re.I),
    ],
    "doxxing": [
        re.compile(r"\bi(?:'ll| will)?\s*(?:post|leak|share|dox)\s+your\s+(?:address|phone|info|details)\b", re.I),
        re.compile(r"\bi know where you (?:live|work)\b", re.I),
        re.compile(r"\bi found your (?:real )?(?:name|address|workplace|family)\b", re.I),
        re.compile(r"\beveryone will know who you (?:really )?are\b", re.I),
    ],
    "dehumanization": [
        re.compile(r"\b(\w+)\s+are\s+not\s+(?:human|people|person)s?\b", re.I),
        re.compile(r"\b(\w+)\s+are\s+animals\b", re.I),
        re.compile(r"\b(\w+)\s+(?:should be|deserve to be)\s+exterminated\b", re.I),
        re.compile(r"\b(\w+)\s+are\s+a\s+disease\b", re.I),
    ],
    "harassment": [
        re.compile(r"\b(?:everyone|let'?s all)\s+(?:report|attack|raid|target)\b(?=\s+@?\w+)", re.I),
        re.compile(r"\bmass report (?:this|that) (?:account|user|person)\b", re.I),
        re.compile(r"\b(?:go after|raid)\s+(?:their\s+(?:profile|account)|@?\w+)\b", re.I),
    ],
}


def regex_filter(text: str) -> tuple[str | None, str | None]:
    for category, patterns in BLOCKLIST.items():
        for pat in patterns:
            if pat.search(text):
                return category, pat.pattern
    return None, None


# ---------------------------------------------------------------------------
# Layer 2: calibrated DistilBERT
# ---------------------------------------------------------------------------
class _Scorer:
    """Wraps a fine-tuned DistilBERT + isotonic calibrator into a callable."""

    def __init__(self, model, tokenizer, isotonic: IsotonicRegression,
                 max_len: int = 128, batch_size: int = 64):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.isotonic = isotonic
        self.max_len = max_len
        self.batch_size = batch_size

    def __call__(self, text: str) -> float:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            enc = self.tokenizer([text], truncation=True, padding=True,
                                 max_length=self.max_len, return_tensors="pt").to(device)
            logits = self.model(**enc).logits
            raw = float(torch.softmax(logits, dim=-1)[0, 1].cpu().numpy())
        return float(self.isotonic.predict([raw])[0])


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class ModerationPipeline:
    """Three-layer guardrail. `predict(text)` returns a decision dict."""

    def __init__(self, prob_fn: Callable[[str], float],
                 review_band: tuple[float, float] = (0.4, 0.6)):
        self.prob_fn = prob_fn
        self.low, self.high = review_band

    def predict(self, comment_text: str) -> dict:
        category, _ = regex_filter(comment_text)
        if category is not None:
            return {"decision": "block", "confidence": 1.0,
                    "triggered_layer": 1, "category": category}

        prob = float(self.prob_fn(comment_text))
        if prob >= self.high:
            decision = "block"
        elif prob <= self.low:
            decision = "pass"
        else:
            decision = "review"
        return {"decision": decision, "confidence": prob,
                "triggered_layer": 3 if decision == "review" else 2}

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, isotonic_path: str,
                        review_band: tuple[float, float] = (0.4, 0.6),
                        device: str | None = None) -> "ModerationPipeline":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
        with open(isotonic_path, "rb") as f:
            isotonic = pickle.load(f)
        return cls(_Scorer(model, tokenizer, isotonic), review_band=review_band)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/baseline")
    p.add_argument("--isotonic", default="checkpoints/baseline/isotonic.pkl")
    p.add_argument("text", nargs="+")
    args = p.parse_args()
    pipe = ModerationPipeline.from_pretrained(args.checkpoint, args.isotonic)
    print(pipe.predict(" ".join(args.text)))
