import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "sample sets" / "sample_20260401.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs_v4_MT"
VALID_LABELS = ("TRUE", "FALSE", "UNCERTAIN")
SUBJECT_TYPES = ("说话人", "第三方", "无", "speaker", "third_party", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate factivity labels on the FIE2026 sample set with a multi-turn prompt."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to the dataset JSON file. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Model name for the OpenAI-compatible chat completions API. Default: gpt-5.4-mini",
    )
    parser.add_argument(
        "--api-key",
        default="sk-An04fTe5nxf2aIhxt6ZDBzzmZci90EPBex3zKKaN0VDVeLMR",
        help="OpenAI API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.chatanywhere.tech/v1",
        help="Base URL for an OpenAI-compatible API. Defaults to OPENAI_BASE_URL or OPENAI_API_BASE.",
    )
    parser.add_argument(
        "--prompt-lang",
        choices=("zh", "en"),
        default="zh",
        help="Prompt language. Choices: zh, en. Default: zh",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "mock"),
        default="openai",
        help="Inference backend. 'openai' uses the OpenAI-compatible chat completions API. Use 'mock' for a local smoke test.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient API errors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for result files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each prediction as it is produced.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_label(value: str) -> str:
    label = value.strip().upper()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {value!r}")
    return label


def normalize_subject_type(value: str) -> str:
    value = value.strip()
    mapping = {
        "说话人": "说话人",
        "speaker": "说话人",
        "第三方": "第三方",
        "third_party": "第三方",
        "无": "无",
        "none": "无",
    }
    if value not in mapping:
        raise ValueError(f"Invalid subject_type: {value!r}")
    return mapping[value]


def extract_tag(raw_text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", raw_text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract <{tag}> from model output: {raw_text!r}")
    return match.group(1).strip()


def extract_answer_label(raw_text: str) -> str:
    match = re.search(r"<answer>\s*(TRUE|FALSE|UNCERTAIN)\s*</answer>", raw_text, re.IGNORECASE)
    if match:
        return normalize_label(match.group(1))
    bare = raw_text.strip()
    if bare.upper() in VALID_LABELS:
        return normalize_label(bare)
    raise ValueError(f"Could not extract label from model output: {raw_text!r}")


def parse_extraction_output(raw_text: str) -> dict[str, str]:
    subject_type = normalize_subject_type(extract_tag(raw_text, "subject_type"))
    proposition_subject = extract_tag(raw_text, "proposition_subject")
    attitude_predicate = extract_tag(raw_text, "attitude_predicate")
    attitude_hint = extract_tag(raw_text, "attitude_hint")
    basis = extract_tag(raw_text, "basis")
    return {
        "subject_type": subject_type,
        "proposition_subject": proposition_subject,
        "attitude_predicate": attitude_predicate,
        "attitude_hint": attitude_hint,
        "basis": basis,
    }


def build_first_turn_prompt(text: str, hypothesis: str, prompt_lang: str) -> str:
    if prompt_lang == "en":
        return f"""Task: Extract a small set of intermediate fields for factivity reasoning from the given text and hypothesis.

You are not making the final TRUE/FALSE/UNCERTAIN decision yet.
You only need to extract the fields below so that a later step can infer the speaker's stance.

Field definitions:
1. subject_type:
   - speaker: the relevant subject in the text is the speaker or narrator
   - third_party: the relevant subject in the text is someone other than the speaker
   - none: the relevant subject is absent or not explicitly stated
2. proposition_subject:
   - the subject of the proposition expressed by the hypothesis
   - if absent, output none
3. attitude_predicate:
   - the key trigger expression that carries the relevant subject's tendency toward the proposition
   - do not extract just any ordinary predicate; extract the expression most useful for judging the proposition's semantic direction
4. attitude_hint:
   - a short semantic hint about the trigger expression in this sentence
   - do not give the final label
   - instead describe the directional cue, such as:
     "strong factive trigger", "weak guess", "contains negative connotation", "implies unreliability", "presupposes truth", "supported suspicion"
5. basis:
   - the factual basis, source, evidence, authority, observation, investigation result, correction, or grounding mentioned in the text
   - if no such basis is given, output none

Output format:
You must strictly follow this format and output nothing else:
<think>your analysis</think>
<subject_type>speaker</subject_type>
<proposition_subject>...</proposition_subject>
<attitude_predicate>...</attitude_predicate>
<attitude_hint>...</attitude_hint>
<basis>...</basis>

text: {text}
hypothesis: {hypothesis}"""

    return f"""任务：从给定的 text 和 hypothesis 中提取一组中间字段，用于后续叙实性判断。

这一步不要直接做 TRUE/FALSE/UNCERTAIN 的最终判断。
你只需要提取下面这些字段，供下一步推断说话人的倾向。

字段定义：
1. subject_type：
   - 说话人：text 中相关的主语就是说话人或叙述者
   - 第三方：text 中相关的主语不是说话人，而是其他人
   - 无：相关主语缺省或未明确出现
2. proposition_subject：
   - hypothesis 所表达命题的主语
   - 如果没有明确主语，输出 无
3. attitude_predicate：
   - text 中最能体现相关主语对该命题倾向的关键触发表达
   - 不是随便抽一个普通谓语，而是要抽最有助于后续判断命题方向的表达
4. attitude_hint：
   - 对 attitude_predicate 的简短语义提示
   - 不要直接给最终标签
   - 只说明它体现出的倾向线索，例如：
     “强事实触发词”“弱猜测”“带贬义，暗示不可靠”“预设命题为真”“有根据的怀疑”
5. basis：
   - text 中出现的事实根据、来源、证据、权威、观察、调查结果、纠正信息或其他支撑
   - 如果没有这类根据，输出 无

输出要求：
请严格按照以下格式输出，不要输出其他格式：
<think>你的分析</think>
<subject_type>说话人</subject_type>
<proposition_subject>...</proposition_subject>
<attitude_predicate>...</attitude_predicate>
<attitude_hint>...</attitude_hint>
<basis>...</basis>

text: {text}
hypothesis: {hypothesis}"""


def build_second_turn_prompt(
    text: str,
    hypothesis: str,
    extraction: dict[str, str],
    prompt_lang: str,
) -> str:
    if prompt_lang == "en":
        return f"""Task: Use the extracted fields and the original text to determine the final factivity label of the hypothesis.

The extracted fields below are expert-structured information. You should rely mainly on them.
Use the original text only as a secondary reference for verification. Do not ignore the extracted fields and start over from scratch.

What matters is the speaker's tendency toward the proposition in the hypothesis.

Follow these rules:
1. First check whether the relevant subject is the cognitive subject.
   If subject_type is speaker, judge directly from that subject's tendency.
2. If subject_type is not speaker, inspect the stance attributed to that subject in the sentence.
   If it is only unsupported guessing, believing, hoping, worrying, imagining, or similar weak mentality, and basis is none, output UNCERTAIN.
3. Otherwise, if the sentence provides factual basis, source, evidence, observation, investigation result, correction, authority, or other grounding, treat that as support for the speaker's implicit tendency.
4. Use attitude_predicate and especially attitude_hint to judge the semantic direction carried by the trigger expression.
5. Then combine subject_type, attitude_predicate, attitude_hint, basis, and the original sentence to decide whether the speaker's tendency is positive, negative, or uncertain.

Decision rule:
- positive tendency -> TRUE
- negative tendency -> FALSE
- unsupported uncertainty -> UNCERTAIN

Output format:
You must strictly follow this format and output nothing else:
<think>your analysis</think>
<answer>TRUE</answer>

The answer must be exactly one of: TRUE, FALSE, UNCERTAIN.

Original text: {text}
Hypothesis: {hypothesis}

Extracted fields:
subject_type: {extraction["subject_type"]}
proposition_subject: {extraction["proposition_subject"]}
attitude_predicate: {extraction["attitude_predicate"]}
attitude_hint: {extraction["attitude_hint"]}
basis: {extraction["basis"]}"""

    return f"""任务：结合中间抽取结果和原始 text，判断 hypothesis 的最终叙实性标签。

下面的抽取结果是专家提取的结构化信息，你应当主要参考这份信息。
原始 text 只作为辅助核对材料，不要忽略抽取结果后重新从头自由发挥。

最重要的是判断说话人对 hypothesis 所表达命题的倾向。

请按照以下规则判断：
1. 先看相关主语是不是认知主体。
   如果 subject_type 是“说话人”，就优先根据该主语的倾向判断。
2. 如果 subject_type 不是“说话人”，就看句中归属于该主语的立场。
   如果只是没有事实根据、来源或证据支撑的猜测、认为、希望、担心、幻想等弱心理态度，且 basis 为“无”，则判为 UNCERTAIN。
3. 否则，如果句中给出了事实根据、来源、证据、观察、调查结果、纠正信息、权威信息或其他支撑，就把这些根据视为说话人隐含倾向的支撑。
4. 判断 attitude_predicate 的方向时，要特别参考 attitude_hint 提供的语义提示。
5. 最后结合 subject_type、attitude_predicate、attitude_hint、basis 以及原句整体语义，判断说话人的倾向究竟是正向、反向还是不确定。

判断规则：
- 正向倾向 -> TRUE
- 反向倾向 -> FALSE
- 没有真实承诺的不确定倾向 -> UNCERTAIN

输出要求：
请严格按照以下格式输出，不要输出其他格式：
<think>你的分析</think>
<answer>TRUE</answer>

其中 answer 只能是 TRUE、FALSE、UNCERTAIN 三者之一。

原始 text: {text}
hypothesis: {hypothesis}

抽取结果：
subject_type: {extraction["subject_type"]}
proposition_subject: {extraction["proposition_subject"]}
attitude_predicate: {extraction["attitude_predicate"]}
attitude_hint: {extraction["attitude_hint"]}
basis: {extraction["basis"]}"""


@dataclass
class Prediction:
    sample_id: str
    text: str
    hypothesis: str
    gold: str
    pred: str
    ok: bool
    first_turn_prompt: str
    first_turn_output: str
    extraction: dict[str, str]
    second_turn_prompt: str
    second_turn_output: str


class MockMultiTurnClient:
    def predict(self, text: str, hypothesis: str, prompt_lang: str) -> dict[str, Any]:
        del prompt_lang
        first_turn_prompt = build_first_turn_prompt(text, hypothesis, "zh")
        subject_type = "第三方"
        proposition_subject = "无"
        attitude_predicate = "无"
        attitude_hint = "无"
        basis = "无"

        if text.startswith(("我", "我们")):
            subject_type = "说话人"
        if any(token in text for token in ("他", "她", "他们", "父亲", "观众", "警方", "哥伦布", "约翰", "秦始皇")):
            subject_type = "第三方"

        if any(token in hypothesis for token in ("他", "她", "他们", "父亲", "主人公", "哥伦布", "约翰")):
            proposition_subject = hypothesis[: min(len(hypothesis), 12)]

        markers = (
            "知道", "发现", "意识到", "注意到", "认为", "猜测", "担心", "推测", "估计",
            "错误地认为", "假装", "吹嘘", "控告", "梦想"
        )
        for marker in markers:
            if marker in text:
                attitude_predicate = marker
                break

        hint_map = {
            "知道": "强事实触发词",
            "发现": "强事实触发词",
            "意识到": "预设命题为真",
            "注意到": "预设命题为真",
            "认为": "弱判断，方向未完全定死",
            "猜测": "弱猜测",
            "担心": "弱心理态度，带负向担忧",
            "推测": "推断性表达，需结合根据判断",
            "估计": "弱推断，存在倾向但不强",
            "错误地认为": "显式反向触发",
            "假装": "暗示表面陈述不可靠",
            "吹嘘": "带贬义，暗示不可靠",
            "控告": "指控性表达，需结合根据判断",
            "梦想": "弱心理态度，缺乏现实承诺",
        }
        if attitude_predicate in hint_map:
            attitude_hint = hint_map[attitude_predicate]

        basis_markers = (
            "根据", "通过", "结果", "账本", "监控", "指纹", "观察", "审计", "专家", "宣布", "事实已经证明"
        )
        for marker in basis_markers:
            if marker in text:
                basis = marker
                break

        first_turn_output = (
            "<think>mock extraction</think>"
            f"<subject_type>{subject_type}</subject_type>"
            f"<proposition_subject>{proposition_subject}</proposition_subject>"
            f"<attitude_predicate>{attitude_predicate}</attitude_predicate>"
            f"<attitude_hint>{attitude_hint}</attitude_hint>"
            f"<basis>{basis}</basis>"
        )
        extraction = parse_extraction_output(first_turn_output)
        second_turn_prompt = build_second_turn_prompt(text, hypothesis, extraction, "zh")

        label = "TRUE"
        if extraction["basis"] == "无" and extraction["attitude_predicate"] in {"认为", "猜测", "担心", "估计", "梦想"}:
            label = "UNCERTAIN"
        if extraction["attitude_predicate"] in {"错误地认为", "假装", "吹嘘"}:
            label = "FALSE"
        if extraction["attitude_predicate"] in {"知道", "发现", "意识到", "注意到"}:
            label = "TRUE"

        second_turn_output = f"<think>mock decision</think><answer>{label}</answer>"
        return {
            "first_turn_prompt": first_turn_prompt,
            "first_turn_output": first_turn_output,
            "extraction": extraction,
            "second_turn_prompt": second_turn_prompt,
            "second_turn_output": second_turn_output,
            "pred_label": extract_answer_label(second_turn_output),
        }


class OpenAICompatibleMultiTurnClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None,
        max_retries: int,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "The 'openai' package is required for --provider openai. "
                "Install it with: pip install openai"
            ) from exc

        self.model = model
        self.max_retries = max_retries
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**client_kwargs)

    def chat_once(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Model returned empty content.")
                return content.strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise RuntimeError(f"Chat completion request failed: {exc}") from exc
        raise RuntimeError(f"Prediction failed after retries: {last_error}")

    def predict(self, text: str, hypothesis: str, prompt_lang: str) -> dict[str, Any]:
        first_turn_prompt = build_first_turn_prompt(text, hypothesis, prompt_lang)
        first_messages = [{"role": "user", "content": first_turn_prompt}]
        first_turn_output = self.chat_once(first_messages, max_tokens=256)
        extraction = parse_extraction_output(first_turn_output)
        second_turn_prompt = build_second_turn_prompt(text, hypothesis, extraction, prompt_lang)

        second_messages = [
            {"role": "user", "content": first_turn_prompt},
            {"role": "assistant", "content": first_turn_output},
            {"role": "user", "content": second_turn_prompt},
        ]
        second_turn_output = self.chat_once(second_messages, max_tokens=256)
        pred_label = extract_answer_label(second_turn_output)
        return {
            "first_turn_prompt": first_turn_prompt,
            "first_turn_output": first_turn_output,
            "extraction": extraction,
            "second_turn_prompt": second_turn_prompt,
            "second_turn_output": second_turn_output,
            "pred_label": pred_label,
        }


def make_client(args: argparse.Namespace) -> Any:
    if args.provider == "mock":
        return MockMultiTurnClient()
    if not args.api_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY or pass --api-key, or use --provider mock."
        )
    return OpenAICompatibleMultiTurnClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
    )


def evaluate(
    data: list[dict[str, Any]],
    client: Any,
    prompt_lang: str,
    sleep_seconds: float,
    verbose: bool,
) -> tuple[list[Prediction], dict[str, Any]]:
    predictions: list[Prediction] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for index, item in enumerate(data, start=1):
        result = client.predict(item["text"], item["hypothesis"], prompt_lang)
        pred_label = normalize_label(result["pred_label"])
        gold_label = normalize_label(item["factivity"])
        ok = pred_label == gold_label

        predictions.append(
            Prediction(
                sample_id=item["id"],
                text=item["text"],
                hypothesis=item["hypothesis"],
                gold=gold_label,
                pred=pred_label,
                ok=ok,
                first_turn_prompt=result["first_turn_prompt"],
                first_turn_output=result["first_turn_output"],
                extraction=result["extraction"],
                second_turn_prompt=result["second_turn_prompt"],
                second_turn_output=result["second_turn_output"],
            )
        )
        confusion[gold_label][pred_label] += 1

        if verbose:
            print(
                f"[{index:03d}] {item['id']} gold={gold_label} pred={pred_label} ok={ok}",
                flush=True,
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    total = len(predictions)
    correct = sum(1 for item in predictions if item.ok)
    accuracy = correct / total if total else 0.0

    per_label: dict[str, dict[str, Any]] = {}
    for label in VALID_LABELS:
        label_items = [item for item in predictions if item.gold == label]
        label_total = len(label_items)
        label_correct = sum(1 for item in label_items if item.ok)
        per_label[label] = {
            "total": label_total,
            "correct": label_correct,
            "accuracy": (label_correct / label_total) if label_total else 0.0,
        }

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_label": per_label,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }
    return predictions, summary


def save_results(
    output_dir: Path,
    dataset_path: Path,
    provider: str,
    model: str,
    prompt_lang: str,
    predictions: list[Prediction],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"factivity_label_multiturn_eval_{provider}_{model}_{timestamp}.json"
    errors_path = output_dir / f"factivity_label_multiturn_errors_{provider}_{model}_{timestamp}.json"

    payload = {
        "dataset": str(dataset_path),
        "provider": provider,
        "model": model,
        "prompt_lang": prompt_lang,
        "summary": summary,
        "predictions": [
            {
                "id": item.sample_id,
                "text": item.text,
                "hypothesis": item.hypothesis,
                "gold": item.gold,
                "pred": item.pred,
                "ok": item.ok,
                "first_turn_prompt": item.first_turn_prompt,
                "extraction": item.extraction,
                "first_turn_output": item.first_turn_output,
                "second_turn_prompt": item.second_turn_prompt,
                "second_turn_output": item.second_turn_output,
            }
            for item in predictions
        ],
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    error_payload = {
        "dataset": str(dataset_path),
        "provider": provider,
        "model": model,
        "prompt_lang": prompt_lang,
        "summary": summary,
        "errors": [
            {
                "id": item.sample_id,
                "text": item.text,
                "hypothesis": item.hypothesis,
                "gold": item.gold,
                "pred": item.pred,
                "ok": item.ok,
                "first_turn_prompt": item.first_turn_prompt,
                "extraction": item.extraction,
                "first_turn_output": item.first_turn_output,
                "second_turn_prompt": item.second_turn_prompt,
                "second_turn_output": item.second_turn_output,
            }
            for item in predictions
            if not item.ok
        ],
    }
    errors_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path, errors_path


def print_summary(summary: dict[str, Any], result_path: Path, errors_path: Path) -> None:
    print(f"accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
    for label in VALID_LABELS:
        stats = summary["per_label"][label]
        print(f"{label}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    print("confusion:")
    for gold in VALID_LABELS:
        row = summary["confusion"].get(gold, {})
        print(f"  gold={gold}: {row}")
    print(f"saved: {result_path}")
    print(f"errors: {errors_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    data = load_dataset(args.dataset)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    client = make_client(args)
    predictions, summary = evaluate(
        data=data,
        client=client,
        prompt_lang=args.prompt_lang,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
    )
    result_path, errors_path = save_results(
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        provider=args.provider,
        model=args.model.replace("/", "_"),
        prompt_lang=args.prompt_lang,
        predictions=predictions,
        summary=summary,
    )
    print_summary(summary, result_path, errors_path)


if __name__ == "__main__":
    main()
