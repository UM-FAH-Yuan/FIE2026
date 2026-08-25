import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "sample sets" / "sample_20260401.json"
DEFAULT_OUTPUT = ROOT / "sample sets" / "sample_20260401_with_tiers.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map sample-set factivity/confidence labels into tier fields."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input sample-set JSON. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON with tier fields added. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def map_factivity_tier(factivity: str) -> str:
    normalized = factivity.strip().upper()
    mapping = {
        "TRUE": "正叙实",
        "FALSE": "反叙实",
        "UNCERTAIN": "非叙实",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported factivity label: {factivity!r}")
    return mapping[normalized]


def map_confidence_tier(factivity: str, confidence: float) -> str:
    normalized = factivity.strip().upper()
    if normalized == "UNCERTAIN":
        if confidence != 0.5:
            raise ValueError(
                f'UNCERTAIN item must have confidence 0.5, got {confidence!r}'
            )
        return "非叙实"

    if not (0.5 < confidence <= 1.0):
        raise ValueError(
            f"TRUE/FALSE item must have confidence in (0.5, 1.0], got {confidence!r}"
        )

    if confidence > 0.875:
        return "强"
    if confidence > 0.75:
        return "较强"
    if confidence > 0.625:
        return "较弱"
    return "弱"


def add_tier_fields(item: dict[str, Any]) -> dict[str, Any]:
    factivity = str(item["factivity"])
    confidence = float(item["confidence"])
    enriched = dict(item)
    enriched["factivity_tier"] = map_factivity_tier(factivity)
    enriched["confidence_tier"] = map_confidence_tier(factivity, confidence)
    return enriched


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list.")

    enriched = [add_tier_fields(item) for item in data]
    args.output.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {args.output}")
    print(f"items: {len(enriched)}")


if __name__ == "__main__":
    main()
