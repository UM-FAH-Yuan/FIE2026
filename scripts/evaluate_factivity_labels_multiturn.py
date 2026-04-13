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
DEFAULT_OUTPUT_DIR = ROOT / "outputs_v6_MT_fine_grained_expert_guidance"
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


def build_expert_guidance(extraction: dict[str, str], prompt_lang: str) -> dict[str, str]:
    subject_type = extraction["subject_type"]
    basis = extraction["basis"].strip()
    basis_is_empty = basis in {"", "无", "none", "None"}

    if prompt_lang == "en":
        if subject_type != "第三方":
            return {
                "expert_rule": "SPEAKER_MAIN_VIEW",
                "expert_advice": (
                    "Expert advice: The main viewpoint of this sentence should be treated as the "
                    "**speaker**'s viewpoint. Judge the tendency of the **speaker** toward the "
                    "hypothesis directly from the whole sentence."
                ),
            }
        if basis_is_empty:
            return {
                "expert_rule": "THIRD_PARTY_NO_BASIS",
                "expert_advice": (
                    "Expert advice: The relevant subject is not the **speaker**, and there is **no basis** "
                    "in the sentence. This often means that the **speaker** is only reporting another "
                    "person's tendency toward the hypothesis. Our task is to judge the tendency of the "
                    "**speaker**, not the third party. If the wording of the **speaker** does not itself "
                    "carry a clear extra tendency, do not directly convert the third party's tendency into "
                    "the **speaker**'s tendency."
                ),
            }
        return {
            "expert_rule": "THIRD_PARTY_WITH_BASIS",
            "expert_advice": (
                "Expert advice: The relevant subject is not the **speaker**, but there **is basis** in the "
                "sentence. Usually, if the **speaker** did not accept the basis to some extent, the "
                "**speaker** would not provide it. Therefore consider this reasoning chain: the **speaker** "
                "accepts or relies on the basis -> the basis carries a factual direction -> the **speaker** "
                "has a tendency toward the hypothesis. Do not stop at the third party's attitude alone."
            ),
        }

    if subject_type != "第三方":
        return {
            "expert_rule": "SPEAKER_MAIN_VIEW",
            "expert_advice": (
                "专家建议：当前句子的主视角按**说话人**处理。请直接根据整句话中**说话人**对 "
                "hypothesis 的倾向判断，不要改成判断其他主体的倾向。"
            ),
        }
    if basis_is_empty:
        return {
            "expert_rule": "THIRD_PARTY_NO_BASIS",
            "expert_advice": (
                "专家建议：当前相关主语不是**说话人**，且句中**无basis**。这通常意味着**说话人**只是在陈述他人"
                "对 hypothesis 的倾向，而我们的任务是判断**说话人**对 hypothesis 的倾向。如果**说话人**的"
                "表述本身没有额外带出明确倾向，则不能直接把第三方的倾向当成**说话人**的倾向。"
            ),
        }
    return {
        "expert_rule": "THIRD_PARTY_WITH_BASIS",
        "expert_advice": (
            "专家建议：当前相关主语不是**说话人**，但句中**有basis**。请注意，如果**说话人**完全不相信这条"
            "倾向，通常不会主动给出 basis。因此判断时应考虑这条逻辑链：**说话人**相信或依赖 basis -> "
            "basis 本身带有事实倾向 -> **说话人**对 hypothesis 有倾向。不要只停留在第三方态度本身。"
        ),
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
    expert_guidance: dict[str, str],
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
basis: {extraction["basis"]}

Expert rule: {expert_guidance["expert_rule"]}
Expert advice:
{expert_guidance["expert_advice"]}"""

    return f"""任务：结合中间抽取结果和原始 text，判断 hypothesis 的最终叙实性标签。

下面的抽取结果是专家提取的结构化信息，你应当主要参考这份信息。
原始 text 只作为辅助核对材料，不要忽略抽取结果后重新从头自由发挥。

最重要的是判断说话人对 hypothesis 所表达命题的倾向。

请按照以下规则判断：
1. 先看相关主语是不是认知主体。
   如果 subject_type 是“说话人”，就优先根据该主语的倾向判断。
2. 如果 subject_type 不是“说话人”，就看句中归属于该主语的立场。
   如果只是没有事实根据、来源或证据支撑的猜测、认为、希望、担心等弱心理态度，且 basis 为“无”，则判为 UNCERTAIN。
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
basis: {extraction["basis"]}

专家规则：{expert_guidance["expert_rule"]}
专家建议：
{expert_guidance["expert_advice"]}"""


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
    expert_guidance: dict[str, str]
    second_turn_prompt: str
    second_turn_output: str


class MockMultiTurnClient:
    def predict(self, text: str, hypothesis: str, prompt_lang: str) -> dict[str, Any]:
        first_turn_prompt = build_first_turn_prompt(text, hypothesis, prompt_lang)
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
        expert_guidance = build_expert_guidance(extraction, prompt_lang)
        second_turn_prompt = build_second_turn_prompt(text, hypothesis, extraction, expert_guidance, prompt_lang)

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
            "expert_guidance": expert_guidance,
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
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content.strip()
                if len(messages)==3:
                    print('----------')
                    print(content)
                    print('----------')
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
        first_turn_output = self.chat_once(first_messages, max_tokens=1280)
        extraction = parse_extraction_output(first_turn_output)
        expert_guidance = build_expert_guidance(extraction, prompt_lang)
        second_turn_prompt = build_second_turn_prompt(text, hypothesis, extraction, expert_guidance, prompt_lang)

        second_messages = [
            {"role": "user", "content": first_turn_prompt},
            {"role": "assistant", "content": first_turn_output},
            {"role": "user", "content": second_turn_prompt},
        ]
        second_turn_output = self.chat_once(second_messages, max_tokens=1280)
        pred_label = extract_answer_label(second_turn_output)
        return {
            "first_turn_prompt": first_turn_prompt,
            "first_turn_output": first_turn_output,
            "extraction": extraction,
            "expert_guidance": expert_guidance,
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


def build_summary(predictions: list[Prediction]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for item in predictions:
        confusion[item.gold][item.pred] += 1

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

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_label": per_label,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }


def prediction_to_dict(item: Prediction) -> dict[str, Any]:
    return {
        "id": item.sample_id,
        "text": item.text,
        "hypothesis": item.hypothesis,
        "gold": item.gold,
        "pred": item.pred,
        "ok": item.ok,
        "first_turn_prompt": item.first_turn_prompt,
        "extraction": item.extraction,
        "expert_guidance": item.expert_guidance,
        "first_turn_output": item.first_turn_output,
        "second_turn_prompt": item.second_turn_prompt,
        "second_turn_output": item.second_turn_output,
    }


def prediction_from_dict(item: dict[str, Any]) -> Prediction:
    return Prediction(
        sample_id=item["id"],
        text=item["text"],
        hypothesis=item["hypothesis"],
        gold=normalize_label(item["gold"]),
        pred=normalize_label(item["pred"]),
        ok=bool(item["ok"]),
        first_turn_prompt=item["first_turn_prompt"],
        first_turn_output=item["first_turn_output"],
        extraction=item["extraction"],
        expert_guidance=item["expert_guidance"],
        second_turn_prompt=item["second_turn_prompt"],
        second_turn_output=item["second_turn_output"],
    )


def list_main_result_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if "_errors_" not in path.name and path.name.startswith("factivity_label_multiturn_eval_")
    )


def resolve_result_paths(output_dir: Path, provider: str, model: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list_main_result_files(output_dir)
    if len(existing) > 1:
        raise SystemExit(
            "Found multiple non-error result JSON files in the output directory. "
            "Keep only one resumable result file before rerunning:\n"
            + "\n".join(str(path) for path in existing)
        )
    if existing:
        result_path = existing[0]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"factivity_label_multiturn_eval_{provider}_{model}_{timestamp}.json"
    errors_path = Path(str(result_path).replace("_eval_", "_errors_"))
    return result_path, errors_path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def save_main_results(
    result_path: Path,
    dataset_path: Path,
    provider: str,
    model: str,
    prompt_lang: str,
    predictions: list[Prediction],
) -> dict[str, Any]:
    summary = build_summary(predictions)
    payload = {
        "dataset": str(dataset_path),
        "provider": provider,
        "model": model,
        "prompt_lang": prompt_lang,
        "summary": summary,
        "predictions": [prediction_to_dict(item) for item in predictions],
    }
    write_json_atomic(result_path, payload)
    return summary


def save_error_results(
    errors_path: Path,
    dataset_path: Path,
    provider: str,
    model: str,
    prompt_lang: str,
    predictions: list[Prediction],
    summary: dict[str, Any],
) -> None:
    error_payload = {
        "dataset": str(dataset_path),
        "provider": provider,
        "model": model,
        "prompt_lang": prompt_lang,
        "summary": summary,
        "errors": [prediction_to_dict(item) for item in predictions if not item.ok],
    }
    write_json_atomic(errors_path, error_payload)


def load_resume_predictions(
    result_path: Path,
    data: list[dict[str, Any]],
    dataset_path: Path,
    provider: str,
    model: str,
    prompt_lang: str,
) -> list[Prediction]:
    if not result_path.exists():
        return []

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Existing result file is not valid JSON and cannot be resumed: {result_path}\n{exc}"
        ) from exc

    saved_dataset = str(payload.get("dataset", ""))
    saved_provider = str(payload.get("provider", ""))
    saved_model = str(payload.get("model", ""))
    saved_prompt_lang = str(payload.get("prompt_lang", ""))
    if (
        saved_dataset and saved_dataset != str(dataset_path)
        or saved_provider and saved_provider != provider
        or saved_model and saved_model != model
        or saved_prompt_lang and saved_prompt_lang != prompt_lang
    ):
        raise SystemExit(
            "Existing result file does not match the current dataset/provider/model/prompt language: "
            f"{result_path}"
        )

    raw_predictions = payload.get("predictions", [])
    if not isinstance(raw_predictions, list):
        raise SystemExit(f"Invalid predictions field in existing result file: {result_path}")

    predictions = [prediction_from_dict(item) for item in raw_predictions]
    dataset_by_id = {item["id"]: item for item in data}

    for index, prediction in enumerate(predictions, start=1):
        if index > len(data):
            raise SystemExit(
                f"Existing result file has more predictions than current dataset subset: {result_path}"
            )
        expected = data[index - 1]
        if prediction.sample_id != expected["id"]:
            raise SystemExit(
                "Existing result file does not align with the current dataset order at "
                f"position {index}: expected {expected['id']}, got {prediction.sample_id}"
            )
        gold_label = normalize_label(expected["factivity"])
        if prediction.gold != gold_label:
            raise SystemExit(
                f"Gold label mismatch for {prediction.sample_id} in existing result file: {result_path}"
            )
        if prediction.text != expected["text"] or prediction.hypothesis != expected["hypothesis"]:
            raise SystemExit(
                f"Text or hypothesis mismatch for {prediction.sample_id} in existing result file: {result_path}"
            )

    return predictions


def evaluate(
    data: list[dict[str, Any]],
    client: Any,
    prompt_lang: str,
    sleep_seconds: float,
    verbose: bool,
    result_path: Path,
    dataset_path: Path,
    provider: str,
    model: str,
    initial_predictions: list[Prediction] | None = None,
) -> tuple[list[Prediction], dict[str, Any]]:
    predictions: list[Prediction] = list(initial_predictions or [])
    start_index = len(predictions)

    for index, item in enumerate(data[start_index:], start=start_index + 1):
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
                expert_guidance=result["expert_guidance"],
                second_turn_prompt=result["second_turn_prompt"],
                second_turn_output=result["second_turn_output"],
            )
        )
        summary = save_main_results(
            result_path=result_path,
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            prompt_lang=prompt_lang,
            predictions=predictions,
        )

        if verbose:
            print(
                f"[{index:03d}] {item['id']} gold={gold_label} pred={pred_label} ok={ok}",
                flush=True,
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return predictions, build_summary(predictions)


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


def _is_third_party_subject(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return (
        normalized == "third_party"
        or "第三" in stripped
        or "third" in normalized
        or "笁鏂" in stripped
    )


def _is_speaker_or_none_subject(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return (
        normalized in {"speaker", "none"}
        or "说话" in stripped
        or "璇磋瘽" in stripped
        or stripped == "无"
        or "鏃" in stripped
    )


def _is_empty_basis(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized in {"", "none"} or stripped == "无" or "鏃" in stripped


def build_expert_guidance(extraction: dict[str, str], prompt_lang: str) -> dict[str, str]:
    subject_type = extraction["subject_type"]
    basis = extraction["basis"]
    basis_is_empty = _is_empty_basis(basis)
    is_third_party = _is_third_party_subject(subject_type)

    if prompt_lang == "en":
        if not is_third_party:
            return {
                "expert_rule": "SPEAKER_MAIN_VIEW",
                "expert_advice": (
                    "Expert advice: The relevant viewpoint here should be treated as the **speaker**'s "
                    "viewpoint. Judge the stance of the **speaker** toward the hypothesis directly from "
                    "the whole sentence. Do not rewrite the task into judging some other entity's stance."
                ),
            }
        if basis_is_empty:
            return {
                "expert_rule": "THIRD_PARTY_NO_BASIS",
                "expert_advice": (
                    "Expert advice: The relevant subject is not the **speaker**, and there is **no basis** "
                    "in the sentence. Do not directly convert the third party's stance into the "
                    "**speaker**'s stance. First check whether the sentence only neutrally reports the "
                    "third party's subjective cognitive activity, or whether the **speaker** adds an extra "
                    "directional signal through evaluation, correction, presupposition, negation, or "
                    "semantic connotation. Only if it is merely neutral reporting with no extra "
                    "directional signal should you consider UNCERTAIN."
                ),
            }
        return {
            "expert_rule": "THIRD_PARTY_WITH_BASIS",
            "expert_advice": (
                "Expert advice: The relevant subject is not the **speaker**, but there **is basis** in the "
                "sentence. This usually means the **speaker** is not merely neutral reporting the third "
                "party's stance, but is introducing information that supports some direction. Treat the "
                "basis as an important clue for the **speaker**'s implicit stance, and combine it with "
                "attitude_predicate, attitude_hint, and the whole sentence to decide the **speaker**'s "
                "direction toward the hypothesis."
            ),
        }

    if not is_third_party:
        return {
            "expert_rule": "SPEAKER_MAIN_VIEW",
            "expert_advice": (
                "专家建议：当前相关视角按**说话人**处理。请直接根据整句话中**说话人**对 hypothesis "
                "的表达来判断，不要把任务改写成判断其他主体的立场。"
            ),
        }
    if basis_is_empty:
        return {
            "expert_rule": "THIRD_PARTY_NO_BASIS",
            "expert_advice": (
                "专家建议：当前相关主语不是**说话人**，且句中**无basis**。请不要直接把第三方态度当成"
                "**说话人**的态度。先判断这句话是否只是中性报告第三方的主观认知活动，还是**说话人**通过"
                "评价、纠正、预设、否定或语义褒贬等表达额外带出了自己的方向性信号。只有在确实只是中性转述、"
                "且没有额外方向性信号时，才考虑 UNCERTAIN。"
            ),
        }
    return {
        "expert_rule": "THIRD_PARTY_WITH_BASIS",
        "expert_advice": (
            "专家建议：当前相关主语不是**说话人**，且句中**有basis**。这通常说明**说话人**并非完全中性地"
            "转述第三方态度，而是在引入可支撑某一方向的信息。请把 basis 视为判断**说话人**隐含立场的重要"
            "线索，并结合 attitude_predicate、attitude_hint 和整体语义，一起判断**说话人**对 "
            "hypothesis 的倾向。"
        ),
    }


def build_first_turn_prompt(text: str, hypothesis: str, prompt_lang: str) -> str:
    if prompt_lang == "en":
        return f"""Task: Extract a small set of intermediate fields for factivity reasoning from the given text and hypothesis.

You are not making the final TRUE/FALSE/UNCERTAIN decision yet.
You only need to extract the fields below so that a later step can infer the speaker's stance.

Field definitions:
1. subject_type:
   - speaker: the relevant viewpoint in the text is the speaker or narrator
   - third_party: the relevant viewpoint in the text belongs to someone other than the speaker
   - none: the relevant viewpoint is absent or not explicitly stated
2. proposition_subject:
   - the subject of the proposition expressed by the hypothesis
   - if absent, output none
3. attitude_predicate:
   - the key trigger expression in the sentence that is most useful for judging the **speaker**'s stance toward the hypothesis
   - it can be a direct stance expression by the **speaker**, or an evaluative, corrective, presuppositional, negative, or otherwise directional expression added by the **speaker** while talking about a third party
   - do not extract just any ordinary predicate
4. attitude_hint:
   - a short abstract hint explaining how attitude_predicate affects the judgment of the **speaker**'s stance in this sentence
   - do not give the final label
   - focus on whether it directly expresses the **speaker**'s stance, merely reports a third party's subjective cognitive activity, or lets the **speaker** add evaluation, correction, presupposition, negation, or semantic connotation
5. basis:
   - the factual basis, source, evidence, authority, observation, investigation result, correction, or other grounding introduced in the sentence that may support the inference about the **speaker**'s stance
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

    return f"""任务：从给定的 text 和 hypothesis 中提取一组中间字段，用于后续的述实性判断。

这一步不要直接做 TRUE / FALSE / UNCERTAIN 的最终判断。
你只需要提取下面这些字段，供下一步推断**说话人**对 hypothesis 的态度。

字段定义：
1. subject_type：
   - 说话人：text 中相关视角就是说话人或叙述者
   - 第三方：text 中相关视角不是说话人，而是其他主体
   - 无：相关视角缺省或未明确出现
2. proposition_subject：
   - hypothesis 所表达命题的主语
   - 如果没有明确主语，输出 无
3. attitude_predicate：
   - 句中最关键、最有助于后续判断**说话人**对 hypothesis 立场的触发表达
   - 它既可以是**说话人**直接表达立场的词语，也可以是**说话人**在谈论第三方时额外加入的评价、纠正、预设、否定或其他方向性表达
   - 不要随便抽一个普通谓语
4. attitude_hint：
   - 用一句抽象的话说明 attitude_predicate 在本句中如何影响对**说话人**立场的判断
   - 不要给最终标签
   - 重点说明它是在直接表达**说话人**立场，还是仅在报告第三方主观认知活动，或者**说话人**是否借它额外加入了评价、纠正、预设、否定或语义褒贬等方向性信息
5. basis：
   - 句中出现的、可用于支撑对**说话人**立场推断的事实根据、来源、证据、观察、调查结果、纠正信息、权威信息或其他支撑
   - 如果没有这类信息，输出 无

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
    expert_guidance: dict[str, str],
    prompt_lang: str,
) -> str:
    if prompt_lang == "en":
        return f"""Task: Use the extracted fields and the original text to determine the final factivity label of the hypothesis.

The extracted fields below are expert-structured information. You should rely mainly on them.
Use the original text only as a secondary reference for verification. Do not ignore the extracted fields and start over from scratch.

The final label must always be based on the **speaker**'s stance toward the proposition in the hypothesis, not directly on a third party's stance.

Follow these rules:
1. If subject_type is speaker or none, judge directly from the stance expressed by the **speaker** in the sentence.
2. If subject_type is third_party, do not directly use the third party's stance as the answer.
3. When subject_type is third_party, first check whether the **speaker** adds extra directional information while talking about that third party. Such information may come from evaluation, correction, negation of the third party's cognition, presupposition, semantic connotation, or support reflected in basis.
4. If the sentence merely neutrally reports the third party's subjective cognitive activity, and the **speaker** does not add any extra directional signal, then consider UNCERTAIN.
5. Treat basis as an important clue for whether the **speaker** is supporting some direction, but do not let basis alone decide the label without attitude_predicate, attitude_hint, and the whole sentence.
6. Finally combine subject_type, attitude_predicate, attitude_hint, basis, and the original sentence to decide whether the **speaker**'s stance is positive, negative, or uncertain.

Decision rule:
- positive tendency -> TRUE
- negative tendency -> FALSE
- no directional stance from the **speaker** -> UNCERTAIN

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
basis: {extraction["basis"]}

Expert rule: {expert_guidance["expert_rule"]}
Expert advice:
{expert_guidance["expert_advice"]}"""

    return f"""任务：结合中间抽取结果和原始 text，判断 hypothesis 的最终述实性标签。
下面的抽取结果是专家提取的结构化信息，你应当主要参考这份信息。原始 text 只作为辅助核对材料，不要忽略抽取结果后重新从头自由发挥。

最终标签永远基于**说话人**对 hypothesis 所表达命题的态度，而不是直接基于第三方自己的态度。

请按照以下规则判断：
1. 如果 subject_type 是“说话人”或“无”，就优先根据句中直接体现的**说话人**立场判断。
2. 如果 subject_type 是“第三方”，不要直接把第三方态度当成答案。
3. 当 subject_type 是“第三方”时，先判断**说话人**在转述该第三方态度时，是否额外加入了自己的方向性信息。这些信息可以来自评价、纠正、否定对方认知、预设、语义褒贬，以及 basis 所体现的支撑信息。
4. 如果句子只是中性报告第三方的主观认知活动，而**说话人**没有额外加入任何方向性信号，才考虑判为 UNCERTAIN。
5. basis 是判断**说话人**是否在为某一方向提供支撑的重要线索，但不能脱离 attitude_predicate、attitude_hint 和整体语义单独决定标签。
6. 最后结合 subject_type、attitude_predicate、attitude_hint、basis 以及原句整体语义，判断**说话人**的态度究竟是正向、反向还是不确定。

判断规则：
- 正向倾向 -> TRUE
- 反向倾向 -> FALSE
- **说话人**没有体现出方向性立场 -> UNCERTAIN

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
basis: {extraction["basis"]}

专家规则：{expert_guidance["expert_rule"]}
专家建议：
{expert_guidance["expert_advice"]}"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    data = load_dataset(args.dataset)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    model_name = args.model.replace("/", "_")
    result_path, errors_path = resolve_result_paths(args.output_dir, args.provider, model_name)
    initial_predictions = load_resume_predictions(
        result_path=result_path,
        data=data,
        dataset_path=args.dataset,
        provider=args.provider,
        model=model_name,
        prompt_lang=args.prompt_lang,
    )
    if initial_predictions:
        print(
            f"Resuming from {result_path} with {len(initial_predictions)}/{len(data)} completed samples.",
            flush=True,
        )

    client = make_client(args)
    predictions, summary = evaluate(
        data=data,
        client=client,
        prompt_lang=args.prompt_lang,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
        result_path=result_path,
        dataset_path=args.dataset,
        provider=args.provider,
        model=model_name,
        initial_predictions=initial_predictions,
    )
    summary = save_main_results(
        result_path=result_path,
        dataset_path=args.dataset,
        provider=args.provider,
        model=model_name,
        prompt_lang=args.prompt_lang,
        predictions=predictions,
    )
    save_error_results(
        errors_path=errors_path,
        dataset_path=args.dataset,
        provider=args.provider,
        model=model_name,
        prompt_lang=args.prompt_lang,
        predictions=predictions,
        summary=summary,
    )
    print_summary(summary, result_path, errors_path)


if __name__ == "__main__":
    main()
