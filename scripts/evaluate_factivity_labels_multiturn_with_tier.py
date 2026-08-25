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
DEFAULT_DATASET = ROOT / "sample sets" / "sample_20260401_with_tiers.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs_v9_tiers"
VALID_LABELS = ("TRUE", "FALSE", "UNCERTAIN")
VALID_CONFIDENCE_TIERS = ("弱", "较弱", "较强", "强", "非叙实")
VALID_FINAL_TIERS = (
    "强反叙实",
    "较强反叙实",
    "较弱反叙实",
    "弱反叙实",
    "非叙实",
    "弱正叙实",
    "较弱正叙实",
    "较强正叙实",
    "强正叙实",
)
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


def normalize_confidence_tier(value: str) -> str:
    tier = value.strip().replace("較弱", "较弱").replace("較强", "较强")
    tier_map = {
        "weak": "弱",
        "relatively_weak": "较弱",
        "relatively_strong": "较强",
        "strong": "强",
    }
    tier = tier_map.get(tier.lower(), tier)
    if tier not in VALID_CONFIDENCE_TIERS:
        raise ValueError(f"Invalid confidence_tier: {value!r}")
    return tier


def extract_confidence_tier(raw_text: str) -> str:
    match = re.search(
        r"<confidence_tier>\s*(弱|较弱|較弱|较强|較强|强|非叙实|weak|relatively_weak|relatively_strong|strong)\s*</confidence_tier>",
        raw_text,
        re.IGNORECASE,
    )
    if match:
        return normalize_confidence_tier(match.group(1))
    bare = raw_text.strip()
    if bare in VALID_CONFIDENCE_TIERS or bare in {"較弱", "較强", "weak", "relatively_weak", "relatively_strong", "strong"}:
        return normalize_confidence_tier(bare)
    raise ValueError(f"Could not extract confidence_tier from model output: {raw_text!r}")


def compose_final_tier_label(factivity: str, confidence_tier: str) -> str:
    factivity = normalize_label(factivity)
    confidence_tier = normalize_confidence_tier(confidence_tier)
    if factivity == "UNCERTAIN" or confidence_tier == "非叙实":
        return "非叙实"
    if factivity == "TRUE":
        label = f"{confidence_tier}正叙实"
    else:
        label = f"{confidence_tier}反叙实"
    if label not in VALID_FINAL_TIERS:
        raise ValueError(
            f"Invalid composed final tier label from factivity={factivity!r}, "
            f"confidence_tier={confidence_tier!r}"
        )
    return label


def map_confidence_to_tier(factivity: str, confidence: float | int | str | None) -> str:
    factivity = normalize_label(factivity)
    if factivity == "UNCERTAIN":
        return "非叙实"
    if confidence is None:
        raise ValueError("Missing confidence for non-UNCERTAIN sample.")
    value = float(confidence)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"confidence out of range [0,1]: {confidence!r}")
    if value <= 0.625:
        return "弱"
    if value <= 0.75:
        return "较弱"
    if value <= 0.875:
        return "较强"
    return "强"


def get_gold_confidence_tier(item: dict[str, Any]) -> str | None:
    tier = item.get("confidence_tier")
    if tier is not None:
        return normalize_confidence_tier(tier)
    confidence = item.get("confidence")
    factivity = item.get("factivity")
    if factivity is None or confidence is None:
        return None
    return map_confidence_to_tier(str(factivity), confidence)


def get_gold_final_tier(item: dict[str, Any], gold_factivity: str, gold_confidence_tier: str | None) -> str | None:
    final_tier = item.get("final_tier")
    if final_tier is not None:
        return str(final_tier)
    factivity_tier = item.get("factivity_tier")
    if factivity_tier == "非叙实":
        return "非叙实"
    if gold_confidence_tier is None:
        return None
    return compose_final_tier_label(gold_factivity, gold_confidence_tier)


def log_stage_block(title: str, content: str) -> None:
    print("-" * 80)
    print(f"[{title}]")
    print("-" * 80)
    print(content)


# def build_expert_guidance(extraction: dict[str, str], prompt_lang: str) -> dict[str, str]:
#     subject_type = extraction["subject_type"]
#     basis = extraction["basis"].strip()
#     basis_is_empty = basis in {"", "无", "none", "None"}

#     if prompt_lang == "en":
#         if subject_type != "第三方":
#             return {
#                 "expert_rule": "SPEAKER_MAIN_VIEW",
#                 "expert_advice": (
#                     "Expert advice: The main viewpoint of this sentence should be treated as the "
#                     "**speaker**'s viewpoint. Judge the tendency of the **speaker** toward the "
#                     "hypothesis directly from the whole sentence."
#                 ),
#             }
#         if basis_is_empty:
#             return {
#                 "expert_rule": "THIRD_PARTY_NO_BASIS",
#                 "expert_advice": (
#                     "Expert advice: The relevant subject is not the **speaker**, and there is **no basis** "
#                     "in the sentence. This often means that the **speaker** is only reporting another "
#                     "person's tendency toward the hypothesis. Our task is to judge the tendency of the "
#                     "**speaker**, not the third party. If the wording of the **speaker** does not itself "
#                     "carry a clear extra tendency, do not directly convert the third party's tendency into "
#                     "the **speaker**'s tendency."
#                 ),
#             }
#         return {
#             "expert_rule": "THIRD_PARTY_WITH_BASIS",
#             "expert_advice": (
#                 "Expert advice: The relevant subject is not the **speaker**, but there **is basis** in the "
#                 "sentence. Usually, if the **speaker** did not accept the basis to some extent, the "
#                 "**speaker** would not provide it. Therefore consider this reasoning chain: the **speaker** "
#                 "accepts or relies on the basis -> the basis carries a factual direction -> the **speaker** "
#                 "has a tendency toward the hypothesis. Do not stop at the third party's attitude alone."
#             ),
#         }

#     if subject_type != "第三方":
#         return {
#             "expert_rule": "SPEAKER_MAIN_VIEW",
#             "expert_advice": (
#                 "专家建议：当前句子的主视角按**说话人**处理。请直接根据整句话中**说话人**对 "
#                 "hypothesis 的**倾向**判断，**不要**改成判断其他主体的倾向。"
#             ),
#         }
#     if basis_is_empty:
#         return {
#             "expert_rule": "THIRD_PARTY_NO_BASIS",
#             "expert_advice": (
#                 "专家建议：当前相关主语不是**说话人**，且句中**无basis**。这通常意味着**说话人**只是在陈述他人"
#                 "对 hypothesis 的倾向，而我们的任务是判断**说话人**对 hypothesis 的倾向。如果**说话人**的"
#                 "表述本身没有额外带出明确倾向，则不能直接把第三方的倾向当成**说话人**的倾向。"
#             ),
#         }
#     return {
#         "expert_rule": "THIRD_PARTY_WITH_BASIS",
#         "expert_advice": (
#             "专家建议：当前相关主语不是**说话人**，但句中**有basis**。请注意，如果**说话人**完全不相信这条"
#             "倾向，通常不会主动给出 basis。因此判断时应考虑这条逻辑链：**说话人**相信或依赖 basis -> "
#             "basis 本身带有事实倾向 -> **说话人**对 hypothesis 有倾向。不要只停留在第三方态度本身。"
#         ),
#     }


def _legacy_build_first_turn_prompt_v1(text: str, hypothesis: str, prompt_lang: str) -> str:
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


def _legacy_build_second_turn_prompt_v1(
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
   如果 subject_type 是“说话人”，就优先根据该主语的倾向判断。该判断**不需要**基于事实或其他主体倾向的支撑，注意我们的任务是判断**说话人的倾向**，包括不同强弱程度的主观推测
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


def _legacy_is_third_party_subject_v1(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized == "third_party" or stripped == "第三方" or "第三方" in stripped


def _legacy_is_speaker_or_none_subject_v1(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return (
        normalized in {"speaker", "none"}
        or stripped in {"说话人", "无"}
        or "说话人" in stripped
    )


def _legacy_is_empty_basis_v1(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized in {"", "none"} or stripped == "无"


@dataclass
class Prediction:
    sample_id: str
    text: str
    hypothesis: str
    gold: str
    pred: str
    ok: bool
    gold_confidence_tier: str | None
    pred_confidence_tier: str
    confidence_tier_ok: bool | None
    gold_final_tier: str | None
    pred_final_tier: str
    final_tier_ok: bool | None
    first_turn_prompt: str
    first_turn_output: str
    extraction: dict[str, str]
    expert_guidance: dict[str, str]
    second_turn_prompt: str
    second_turn_output: str
    third_turn_prompt: str
    third_turn_output: str


class LegacyMockMultiTurnClientV1:
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

    def chat_once(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.3,
        top_p: float | None = None,
    ) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if top_p is not None:
                    request_kwargs["top_p"] = top_p
                response = self.client.chat.completions.create(
                    **request_kwargs,
                )
                content = response.choices[0].message.content.strip()
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
        log_stage_block("TURN1 Prompt", first_turn_prompt)
        first_turn_output = self.chat_once(
            first_messages,
            max_tokens=1280,
            temperature=0.0,
            top_p=1.0,
        )
        log_stage_block("TURN1 Output", first_turn_output)
        extraction = parse_extraction_output(first_turn_output)
        expert_guidance = build_expert_guidance(extraction, prompt_lang)
        second_turn_prompt = build_second_turn_prompt(text, hypothesis, extraction, expert_guidance, prompt_lang)
        log_stage_block("TURN2 Prompt", second_turn_prompt)

        second_messages = [
            {"role": "user", "content": first_turn_prompt},
            {"role": "assistant", "content": first_turn_output},
            {"role": "user", "content": second_turn_prompt},
        ]
        second_turn_output = self.chat_once(second_messages, max_tokens=1280)
        log_stage_block("TURN2 Output", second_turn_output)
        pred_label = extract_answer_label(second_turn_output)
        third_turn_prompt = ""
        third_turn_output = "[SKIPPED: second-turn silver truth is UNCERTAIN]"
        pred_confidence_tier = "非叙实"
        if pred_label != "UNCERTAIN":
            third_turn_prompt = build_third_turn_prompt(
                text=text,
                hypothesis=hypothesis,
                extraction=extraction,
                silver_truth=pred_label,
                prompt_lang=prompt_lang,
            )
            log_stage_block("TURN3 Prompt", third_turn_prompt)
            third_messages = [
                {"role": "user", "content": first_turn_prompt},
                {"role": "assistant", "content": first_turn_output},
                {"role": "user", "content": second_turn_prompt},
                {"role": "assistant", "content": second_turn_output},
                {"role": "user", "content": third_turn_prompt},
            ]
            third_turn_output = self.chat_once(third_messages, max_tokens=1280)
            log_stage_block("TURN3 Output", third_turn_output)
            pred_confidence_tier = extract_confidence_tier(third_turn_output)
        else:
            log_stage_block("TURN3 Prompt", "[SKIPPED: second-turn silver truth is UNCERTAIN]")
            log_stage_block("TURN3 Output", third_turn_output)
        return {
            "first_turn_prompt": first_turn_prompt,
            "first_turn_output": first_turn_output,
            "extraction": extraction,
            "expert_guidance": expert_guidance,
            "second_turn_prompt": second_turn_prompt,
            "second_turn_output": second_turn_output,
            "third_turn_prompt": third_turn_prompt,
            "third_turn_output": third_turn_output,
            "pred_label": pred_label,
            "pred_confidence_tier": pred_confidence_tier,
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

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_label": per_label,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }
    tier_predictions = [item for item in predictions if item.gold_confidence_tier is not None]
    if tier_predictions:
        tier_confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for item in tier_predictions:
            tier_confusion[item.gold_confidence_tier][item.pred_confidence_tier] += 1

        tier_total = len(tier_predictions)
        tier_correct = sum(1 for item in tier_predictions if item.confidence_tier_ok)
        per_tier: dict[str, dict[str, Any]] = {}
        for tier in VALID_CONFIDENCE_TIERS:
            tier_items = [item for item in tier_predictions if item.gold_confidence_tier == tier]
            if not tier_items:
                continue
            tier_correct_count = sum(1 for item in tier_items if item.confidence_tier_ok)
            per_tier[tier] = {
                "total": len(tier_items),
                "correct": tier_correct_count,
                "accuracy": tier_correct_count / len(tier_items),
            }

        summary["confidence_tier_summary"] = {
            "total": tier_total,
            "correct": tier_correct,
            "accuracy": (tier_correct / tier_total) if tier_total else 0.0,
            "per_tier": per_tier,
            "confusion": {gold: dict(preds) for gold, preds in tier_confusion.items()},
        }
    final_predictions = [item for item in predictions if item.gold_final_tier is not None]
    if final_predictions:
        final_confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for item in final_predictions:
            final_confusion[item.gold_final_tier][item.pred_final_tier] += 1

        final_total = len(final_predictions)
        final_correct = sum(1 for item in final_predictions if item.final_tier_ok)
        per_final_tier: dict[str, dict[str, Any]] = {}
        for tier in VALID_FINAL_TIERS:
            tier_items = [item for item in final_predictions if item.gold_final_tier == tier]
            if not tier_items:
                continue
            tier_correct_count = sum(1 for item in tier_items if item.final_tier_ok)
            per_final_tier[tier] = {
                "total": len(tier_items),
                "correct": tier_correct_count,
                "accuracy": tier_correct_count / len(tier_items),
            }

        summary["final_tier_summary"] = {
            "total": final_total,
            "correct": final_correct,
            "accuracy": (final_correct / final_total) if final_total else 0.0,
            "per_tier": per_final_tier,
            "confusion": {gold: dict(preds) for gold, preds in final_confusion.items()},
        }
    return summary


def prediction_to_dict(item: Prediction) -> dict[str, Any]:
    return {
        "id": item.sample_id,
        "text": item.text,
        "hypothesis": item.hypothesis,
        "gold": item.gold,
        "pred": item.pred,
        "ok": item.ok,
        "gold_factivity": item.gold,
        "pred_factivity": item.pred,
        "factivity_ok": item.ok,
        "gold_confidence_tier": item.gold_confidence_tier,
        "pred_confidence_tier": item.pred_confidence_tier,
        "confidence_tier_ok": item.confidence_tier_ok,
        "gold_final_tier": item.gold_final_tier,
        "pred_final_tier": item.pred_final_tier,
        "final_tier_ok": item.final_tier_ok,
        "first_turn_prompt": item.first_turn_prompt,
        "extraction": item.extraction,
        "expert_guidance": item.expert_guidance,
        "first_turn_output": item.first_turn_output,
        "second_turn_prompt": item.second_turn_prompt,
        "second_turn_output": item.second_turn_output,
        "third_turn_prompt": item.third_turn_prompt,
        "third_turn_output": item.third_turn_output,
    }


def prediction_from_dict(item: dict[str, Any]) -> Prediction:
    gold_confidence_tier = item.get("gold_confidence_tier")
    if gold_confidence_tier is not None:
        gold_confidence_tier = normalize_confidence_tier(gold_confidence_tier)
    pred_confidence_tier = normalize_confidence_tier(item.get("pred_confidence_tier", "非叙实"))
    confidence_tier_ok = item.get("confidence_tier_ok")
    if confidence_tier_ok is not None:
        confidence_tier_ok = bool(confidence_tier_ok)
    gold_final_tier = item.get("gold_final_tier")
    if gold_final_tier is None and gold_confidence_tier is not None:
        gold_final_tier = compose_final_tier_label(item["gold"], gold_confidence_tier)
    pred_final_tier = item.get("pred_final_tier")
    if pred_final_tier is None:
        pred_final_tier = compose_final_tier_label(item["pred"], pred_confidence_tier)
    final_tier_ok = item.get("final_tier_ok")
    if final_tier_ok is None and gold_final_tier is not None:
        final_tier_ok = normalize_label(item["pred"]) == normalize_label(item["gold"]) and pred_final_tier == gold_final_tier
    if final_tier_ok is not None:
        final_tier_ok = bool(final_tier_ok)
    return Prediction(
        sample_id=item["id"],
        text=item["text"],
        hypothesis=item["hypothesis"],
        gold=normalize_label(item["gold"]),
        pred=normalize_label(item["pred"]),
        ok=bool(item["ok"]),
        gold_confidence_tier=gold_confidence_tier,
        pred_confidence_tier=pred_confidence_tier,
        confidence_tier_ok=confidence_tier_ok,
        gold_final_tier=gold_final_tier,
        pred_final_tier=pred_final_tier,
        final_tier_ok=final_tier_ok,
        first_turn_prompt=item["first_turn_prompt"],
        first_turn_output=item["first_turn_output"],
        extraction=item["extraction"],
        expert_guidance=item["expert_guidance"],
        second_turn_prompt=item["second_turn_prompt"],
        second_turn_output=item["second_turn_output"],
        third_turn_prompt=item.get("third_turn_prompt", ""),
        third_turn_output=item.get("third_turn_output", ""),
    )


def list_main_result_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if "_errors_" not in path.name and path.name.startswith("factivity_label_multiturn_with_tier_eval_")
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
        result_path = output_dir / f"factivity_label_multiturn_with_tier_eval_{provider}_{model}_{timestamp}.json"
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
        derived_gold_confidence_tier = get_gold_confidence_tier(expected)
        if prediction.gold_confidence_tier is None and derived_gold_confidence_tier is not None:
            prediction.gold_confidence_tier = derived_gold_confidence_tier
        if prediction.gold_final_tier is None:
            prediction.gold_final_tier = get_gold_final_tier(expected, prediction.gold, prediction.gold_confidence_tier)
        if prediction.pred_final_tier == "非叙实" and prediction.pred != "UNCERTAIN":
            prediction.pred_final_tier = compose_final_tier_label(prediction.pred, prediction.pred_confidence_tier)
        if prediction.confidence_tier_ok is None and prediction.gold_confidence_tier is not None:
            prediction.confidence_tier_ok = prediction.pred_confidence_tier == prediction.gold_confidence_tier
        if prediction.final_tier_ok is None and prediction.gold_final_tier is not None:
            prediction.final_tier_ok = prediction.ok and prediction.pred_final_tier == prediction.gold_final_tier

    return predictions


def is_complete_result(predictions: list[Prediction], data: list[dict[str, Any]]) -> bool:
    return len(predictions) == len(data)


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
        gold_confidence_tier = get_gold_confidence_tier(item)
        pred_confidence_tier = normalize_confidence_tier(result["pred_confidence_tier"])
        confidence_tier_ok = None
        if gold_confidence_tier is not None:
            confidence_tier_ok = pred_confidence_tier == gold_confidence_tier
        gold_final_tier = get_gold_final_tier(item, gold_label, gold_confidence_tier)
        pred_final_tier = compose_final_tier_label(pred_label, pred_confidence_tier)
        final_tier_ok = None
        if gold_final_tier is not None:
            final_tier_ok = ok and pred_final_tier == gold_final_tier

        predictions.append(
            Prediction(
                sample_id=item["id"],
                text=item["text"],
                hypothesis=item["hypothesis"],
                gold=gold_label,
                pred=pred_label,
                ok=ok,
                gold_confidence_tier=gold_confidence_tier,
                pred_confidence_tier=pred_confidence_tier,
                confidence_tier_ok=confidence_tier_ok,
                gold_final_tier=gold_final_tier,
                pred_final_tier=pred_final_tier,
                final_tier_ok=final_tier_ok,
                first_turn_prompt=result["first_turn_prompt"],
                first_turn_output=result["first_turn_output"],
                extraction=result["extraction"],
                expert_guidance=result["expert_guidance"],
                second_turn_prompt=result["second_turn_prompt"],
                second_turn_output=result["second_turn_output"],
                third_turn_prompt=result["third_turn_prompt"],
                third_turn_output=result["third_turn_output"],
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
            message = (
                f"[{index:03d}] {item['id']} "
                f"gold_factivity={gold_label} pred_factivity={pred_label} factivity_ok={ok}"
            )
            if gold_confidence_tier is not None:
                message += (
                    f" gold_confidence_tier={gold_confidence_tier}"
                    f" pred_confidence_tier={pred_confidence_tier}"
                    f" confidence_tier_ok={confidence_tier_ok}"
                    f" gold_final_tier={gold_final_tier}"
                    f" pred_final_tier={pred_final_tier}"
                    f" final_tier_ok={final_tier_ok}"
                )
            print(message, flush=True)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return predictions, build_summary(predictions)


def print_summary(summary: dict[str, Any], result_path: Path, errors_path: Path) -> None:
    final_tier_summary = summary.get("final_tier_summary")
    if final_tier_summary:
        print(
            "final_tier_accuracy: "
            f"{final_tier_summary['accuracy']:.4f} "
            f"({final_tier_summary['correct']}/{final_tier_summary['total']})"
        )
        for tier, stats in final_tier_summary["per_tier"].items():
            print(f"{tier}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        print("final_tier_confusion:")
        for gold, row in final_tier_summary["confusion"].items():
            print(f"  gold={gold}: {row}")

    print(f"factivity_accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
    for label in VALID_LABELS:
        stats = summary["per_label"][label]
        print(f"{label}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    print("factivity_confusion:")
    for gold in VALID_LABELS:
        row = summary["confusion"].get(gold, {})
        print(f"  gold={gold}: {row}")
    print(f"saved: {result_path}")
    print(f"errors: {errors_path}")


def _legacy_is_third_party_subject_v2(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return (
        normalized == "third_party"
        or "第三" in stripped
        or "third" in normalized
        or "笁鏂" in stripped
    )


def _legacy_is_speaker_or_none_subject_v2(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return (
        normalized in {"speaker", "none"}
        or "说话" in stripped
        or "璇磋瘽" in stripped
        or stripped == "无"
        or "鏃" in stripped
    )


def _legacy_is_empty_basis_v2(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized in {"", "none"} or stripped == "无" or "鏃" in stripped


def _legacy_build_first_turn_prompt_v2(text: str, hypothesis: str, prompt_lang: str) -> str:
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


def _legacy_build_second_turn_prompt_v2(
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


def parse_extraction_output(raw_text: str) -> dict[str, str]:
    subject_type = normalize_subject_type(extract_tag(raw_text, "subject_type"))
    proposition_subject = extract_tag(raw_text, "proposition_subject")
    attitude_predicate = extract_tag(raw_text, "attitude_predicate")
    predicate_type = extract_tag(raw_text, "predicate_type")
    attitude_hint = extract_tag(raw_text, "attitude_hint")
    basis = extract_tag(raw_text, "basis")
    return {
        "subject_type": subject_type,
        "proposition_subject": proposition_subject,
        "attitude_predicate": attitude_predicate,
        "predicate_type": predicate_type,
        "attitude_hint": attitude_hint,
        "basis": basis,
    }


class MockMultiTurnClient:
    def predict(self, text: str, hypothesis: str, prompt_lang: str) -> dict[str, Any]:
        first_turn_prompt = build_first_turn_prompt(text, hypothesis, prompt_lang)
        log_stage_block("TURN1 Prompt", first_turn_prompt)
        subject_type = "third_party"
        proposition_subject = "none"
        attitude_predicate = "none"
        predicate_type = "非叙实"
        attitude_hint = "none"
        basis = "none"

        if text.startswith(("我", "我们")):
            subject_type = "speaker"
        if any(token in text for token in ("他", "她", "他们", "父亲", "观众", "警方", "哥伦布", "约翰", "秦始皇")):
            subject_type = "third_party"

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
            "知道": "直接表明说话人或句中视角对命题持较强认定",
            "发现": "通过发现类表达支持命题方向",
            "意识到": "带有预设，通常隐含命题成立",
            "注意到": "带有预设，通常隐含命题成立",
            "认为": "主观认知判断，本身不等于中性事实",
            "猜测": "推测性认知表达，方向较弱",
            "担心": "情绪或意愿相关态度，常带负向倾向",
            "推测": "推断性表达，需要结合整体语义判断方向",
            "估计": "弱推断，但可能体现说话人的方向性立场",
            "错误地认为": "说话人通过纠正或否定对方认知表达反向立场",
            "假装": "评价性表达，暗示表层说法不可靠",
            "吹嘘": "评价性表达，带褒贬色彩并暗示说法不可靠",
            "控告": "带有指控色彩，需要结合整体语义判断",
            "梦想": "意愿或设想类表达，不直接构成客观事实",
        }
        if attitude_predicate in hint_map:
            attitude_hint = hint_map[attitude_predicate]

        predicate_type_map = {
            "知道": "正叙实",
            "发现": "正叙实",
            "意识到": "正叙实",
            "注意到": "正叙实",
            "认为": "非叙实",
            "猜测": "非叙实",
            "担心": "非叙实",
            "推测": "非叙实",
            "估计": "非叙实",
            "错误地认为": "反叙实",
            "假装": "反叙实",
            "吹嘘": "反叙实",
            "控告": "非叙实",
            "梦想": "非叙实",
        }
        if attitude_predicate in predicate_type_map:
            predicate_type = predicate_type_map[attitude_predicate]

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
            f"<predicate_type>{predicate_type}</predicate_type>"
            f"<attitude_hint>{attitude_hint}</attitude_hint>"
            f"<basis>{basis}</basis>"
        )
        log_stage_block("TURN1 Output", first_turn_output)
        extraction = parse_extraction_output(first_turn_output)
        expert_guidance = build_expert_guidance(extraction, prompt_lang)
        second_turn_prompt = build_second_turn_prompt(text, hypothesis, extraction, expert_guidance, prompt_lang)
        log_stage_block("TURN2 Prompt", second_turn_prompt)

        label = "TRUE"
        if extraction["basis"] in {"none", "无"} and extraction["predicate_type"] == "非叙实":
            label = "UNCERTAIN"
        if extraction["predicate_type"] == "反叙实":
            label = "FALSE"
        if extraction["predicate_type"] == "正叙实":
            label = "TRUE"

        second_turn_output = f"<think>mock decision</think><answer>{label}</answer>"
        log_stage_block("TURN2 Output", second_turn_output)
        third_turn_prompt = ""
        third_turn_output = "[SKIPPED: second-turn silver truth is UNCERTAIN]"
        pred_confidence_tier = "非叙实"
        if label != "UNCERTAIN":
            third_turn_prompt = build_third_turn_prompt(
                text=text,
                hypothesis=hypothesis,
                extraction=extraction,
                silver_truth=label,
                prompt_lang=prompt_lang,
            )
            log_stage_block("TURN3 Prompt", third_turn_prompt)
            if extraction["basis"] not in {"none", "无"}:
                pred_confidence_tier = "较强"
            elif extraction["predicate_type"] == "正叙实":
                pred_confidence_tier = "强"
            elif extraction["predicate_type"] == "反叙实":
                pred_confidence_tier = "较强"
            else:
                pred_confidence_tier = "弱"
            third_turn_output = (
                "<think>mock tier decision</think>"
                f"<confidence_tier>{pred_confidence_tier}</confidence_tier>"
            )
            log_stage_block("TURN3 Output", third_turn_output)
        else:
            log_stage_block("TURN3 Prompt", "[SKIPPED: second-turn silver truth is UNCERTAIN]")
            log_stage_block("TURN3 Output", third_turn_output)
        return {
            "first_turn_prompt": first_turn_prompt,
            "first_turn_output": first_turn_output,
            "extraction": extraction,
            "expert_guidance": expert_guidance,
            "second_turn_prompt": second_turn_prompt,
            "second_turn_output": second_turn_output,
            "third_turn_prompt": third_turn_prompt,
            "third_turn_output": third_turn_output,
            "pred_label": extract_answer_label(second_turn_output),
            "pred_confidence_tier": pred_confidence_tier,
        }




def build_first_turn_prompt(text: str, hypothesis: str, prompt_lang: str) -> str:
    if prompt_lang == "en":
        return f"""Task: Extract a small set of intermediate fields for factivity reasoning from the given text and hypothesis.

You are not making the final TRUE/FALSE/UNCERTAIN decision yet.
You only need to extract the fields below so that a later step can infer the speaker's stance.

A predicate-centered view is important here: identify not only the trigger expression itself, but also what kind of predicate it is, because different predicate types guide the speaker's stance differently.

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
4. predicate_type:
   - classify attitude_predicate into one of the following fixed types:
     positive_factive: the predicate tends to support or presuppose the proposition
     negative_factive: the predicate tends to reject, negate, correct, or undermine the proposition
     non_factive: the predicate mainly expresses attitude, cognition, speculation, emotion, desire, or other non-factive content, and does not by itself settle the proposition
5. attitude_hint:
   - a short abstract hint explaining how attitude_predicate and predicate_type affect the judgment of the **speaker**'s stance in this sentence
   - do not give the final label
   - focus on whether it directly expresses the **speaker**'s stance, merely reports a third party's subjective cognitive activity, or lets the **speaker** add evaluation, correction, presupposition, negation, or semantic connotation
6. basis:
   - the factual basis, source, evidence, authority, observation, investigation result, correction, or other grounding introduced in the sentence that may support the inference about the **speaker**'s stance
   - if no such basis is given, output none

Output format:
You must strictly follow this format and output nothing else:
<think>your analysis</think>
<subject_type>speaker/third_party/none</subject_type>
<proposition_subject>...</proposition_subject>
<attitude_predicate>...</attitude_predicate>
<predicate_type>positive_factive/negative_factive/non_factive</predicate_type>
<attitude_hint>...</attitude_hint>
<basis>...</basis>

text: {text}
hypothesis: {hypothesis}"""

    return f"""任务：从给定的 text 和 hypothesis 中提取一组中间字段，用于后续的蕴含属性判断。

这一步不要直接做 TRUE / FALSE / UNCERTAIN 的最终判断。
你只需要提取下面这些字段，用于后续推断 **说话人** 对 hypothesis 的态度。

请注意，这里的抽取是“谓词中心”的：
你不仅要找出体现立场的关键触发表达，还要判断该表达属于哪一类谓词，因为不同类型的谓词会以不同方式影响 **说话人** 的立场。

字段定义：
1. subject_type：
   - 说话人：text 中相关视角就是说话人或叙述者本身
   - 第三方：text 中相关视角不属于说话人，而属于其他主体
   - 无：相关视角缺省，或 hypothesis 中无明确对应主体
2. proposition_subject：
   - hypothesis 所表达命题的主语
   - 如果没有明确主语，输出 无
3. attitude_predicate：
   - 句中最关键、最有助于判断 **说话人** 对 hypothesis 立场的触发表达
   - 它可以是 **说话人** 直接表达立场的表达，也可以是 **说话人** 在谈论第三方时额外加入的评价、纠正、预设、否定或其他方向性表达
   - 不要抽取普通谓词，要抽取“体现命题倾向的关键触发表达”
4. predicate_type：
   - 判断 attitude_predicate 属于以下哪一种固定谓词类型，只能选一个：
     正叙实：该谓词通常倾向于支持命题成立，或带有预设命题成立的效果
     反叙实：该谓词通常倾向于否定、反驳、纠正或削弱命题成立
     非叙实：该谓词主要表达态度、认知、推测、情绪或愿望，本身不直接决定命题真值
5. attitude_hint：
   - 用一句简短的话说明 attitude_predicate 和 predicate_type 在本句中如何影响对 **说话人** 立场的判断
   - 不要给最终标签
   - 重点说明：它是在直接体现 **说话人** 的立场，还是在中性转述第三方主观活动，还是让 **说话人** 额外带出了评价、纠正、预设、否定、语义褒贬等方向性信息
6. basis：
   - 句中出现的、可用于支持推断 **说话人** 立场的事实根据、来源、证据、观察、调查结果、纠正信息、权威信息或其他支撑
   - 如果没有这类信息，输出 无

输出格式：
请严格按照以下格式输出，不要输出其他内容：
<think>简短分析</think>
<subject_type>说话人/第三方/无</subject_type>
<proposition_subject>...</proposition_subject>
<attitude_predicate>...</attitude_predicate>
<predicate_type>正叙实/反叙实/非叙实</predicate_type>
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
When judging the trigger expression, pay attention to both attitude_predicate itself and its predicate_type.

Follow these rules:
1. If subject_type is speaker or none, judge directly from the **speaker**'s stance in the whole sentence.
2. If subject_type is third_party, do not directly use the third party's stance as the answer.
3. When subject_type is third_party, first ask whether the **speaker** adds extra directional information while reporting that third party. Such information may come from evaluation, correction, negation of the other party's cognition, presupposition, semantic connotation, or basis.
4. Use predicate_type to interpret the direction of attitude_predicate:
   - positive_factive usually supports the hypothesis;
   - negative_factive usually rejects, refutes, or weakens the hypothesis;
   - non_factive usually expresses attitude, cognition, speculation, emotion, or desire, and does not by itself settle truth.
5. predicate_type cannot be used mechanically apart from the **speaker**'s viewpoint:
   - if the main viewpoint is the **speaker**, even a non_factive predicate can still lead to TRUE/FALSE if the sentence shows the **speaker**'s directional stance;
   - if the main viewpoint is not the **speaker**, a non_factive predicate may be only a neutral report of a third party's subjective state, and only then should UNCERTAIN be considered;
   - if the main viewpoint is not the **speaker** and the predicate is positive_factive or negative_factive, ask whether the predicate itself shows that the **speaker** adds support, negation, correction, refutation, or weakening.
6. Treat basis as an important clue for whether the **speaker** supports some direction, but do not let basis alone decide the label apart from attitude_predicate, predicate_type, attitude_hint, and the full sentence.
7. Finally combine subject_type, attitude_predicate, predicate_type, attitude_hint, basis, and the whole sentence to decide whether the **speaker**'s stance is positive, negative, or uncertain.

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
predicate_type: {extraction["predicate_type"]}
attitude_hint: {extraction["attitude_hint"]}
basis: {extraction["basis"]}

Expert rule: {expert_guidance["expert_rule"]}
Expert advice:
{expert_guidance["expert_advice"]}"""

    return f"""任务：结合中间抽取结果和原始 text，判断 hypothesis 的最终蕴含属性标签。

下面的抽取结果是专家提取的结构化信息，你应当主要参考这份信息。
原始 text 只作为辅助核对材料，不要忽略抽取结果后重新从头自由发挥。

最终标签永远基于 **说话人** 对 hypothesis 所表达命题的态度，而不是直接基于第三方自己的态度。
在判断时，要同时参考 attitude_predicate 本身和它所属的 predicate_type。

请按照以下规则判断：
1. 如果 subject_type 是“说话人”或“无”，优先根据整句话中 **说话人** 的立场直接判断。
2. 如果 subject_type 是“第三方”，不要直接把第三方态度当成答案。
3. 当 subject_type 是“第三方”时，先判断 **说话人** 在转述该第三方时，是否额外加入了自己的方向性信息。这些信息可以来自评价、纠正、否定对方认知、预设、语义上的褒贬，以及 basis 所体现的支撑信息。
4. 同时结合 predicate_type 理解 attitude_predicate 的方向：
   - 如果 predicate_type 是“正叙实”，通常说明该表达倾向于支持 hypothesis 成立；
   - 如果 predicate_type 是“反叙实”，通常说明该表达倾向于否定、反驳或削弱 hypothesis；
   - 如果 predicate_type 是“非叙实”，通常说明该表达主要体现态度、认知、推测、情绪或愿望，本身未必直接决定命题真值。
5. predicate_type 不能脱离 **说话人** 视角机械使用：
   - 当主视角按 **说话人** 处理时，即使 predicate_type 是“非叙实”，只要整句话已经体现出 **说话人** 对 hypothesis 的方向性倾向，仍应据此判断，而不是机械地判为 UNCERTAIN。
   - 当主视角不是 **说话人** 时，如果 predicate_type 是“非叙实”，要特别注意这是否只是对第三方主观状态的中性转述；只有在确实只是中性转述，且 **说话人** 没有额外带出方向性信息时，才考虑判为 UNCERTAIN。
   - 当主视角不是 **说话人** 时，如果 predicate_type 是“正叙实”或“反叙实”，则要进一步判断该谓词是否说明 **说话人** 已经借该表达额外带出了支持、否定、纠正、反驳或削弱等方向。
6. basis 是判断 **说话人** 是否在为某一方向提供支撑的重要线索，但不能脱离 attitude_predicate、predicate_type、attitude_hint 和整体语义单独决定标签。
7. 最后结合 subject_type、attitude_predicate、predicate_type、attitude_hint、basis 以及原句整体语义，判断 **说话人** 的态度究竟是正向、反向还是不确定。

判断规则：
- 正向倾向 -> TRUE
- 反向倾向 -> FALSE
- **说话人** 没有体现出方向性立场 -> UNCERTAIN

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
predicate_type: {extraction["predicate_type"]}
attitude_hint: {extraction["attitude_hint"]}
basis: {extraction["basis"]}

专家规则：{expert_guidance["expert_rule"]}
专家建议：
{expert_guidance["expert_advice"]}"""


def build_third_turn_prompt(
    text: str,
    hypothesis: str,
    extraction: dict[str, str],
    silver_truth: str,
    prompt_lang: str,
) -> str:
    if prompt_lang == "en":
        return f"""Task: Based on the given silver truth, judge only the confidence tier for the current sample.

Notes:
1. The second turn has already produced the silver truth, and it is fixed for this sample.
2. You must not re-judge whether the hypothesis is TRUE, FALSE, or UNCERTAIN.
3. Your only task now is to determine which confidence tier best matches the **speaker**'s strength of stance toward the hypothesis under this fixed result.
4. Here, confidence tier means the **speaker**'s degree of tendency toward the hypothesis, not the model's confidence in its own answer.
5. You must choose exactly one of these four confidence tiers:
- weak
- relatively_weak
- relatively_strong
- strong

Current silver truth: {silver_truth}

Judge using:
- the original text
- the hypothesis
- the first-turn extracted fields
- the second-turn silver truth

Emphasis again:
- do not re-judge factivity
- do not output TRUE, FALSE, or UNCERTAIN
- confidence tier here means the graded strength of the **speaker**'s tendency toward the hypothesis under the fixed silver truth, not the model's self-confidence
- output only one confidence tier: weak / relatively_weak / relatively_strong / strong

Output format:
You must strictly follow this format and output nothing else:
<think>brief analysis</think>
<confidence_tier>weak/relatively_weak/relatively_strong/strong</confidence_tier>

Original text: {text}
Hypothesis: {hypothesis}

First-turn extracted fields:
subject_type: {extraction["subject_type"]}
proposition_subject: {extraction["proposition_subject"]}
attitude_predicate: {extraction["attitude_predicate"]}
predicate_type: {extraction["predicate_type"]}
attitude_hint: {extraction["attitude_hint"]}
basis: {extraction["basis"]}

Second-turn result:
silver_truth: {silver_truth}"""

    return f"""任务：在已经给定 silver truth 的基础上，只判断当前样本的 confidence tier。

注意：
1. 第二轮已经给出了 silver truth，它是当前样本的既定判断结果。
2. 你现在不能重新判断 hypothesis 是 TRUE、FALSE 或 UNCERTAIN。
3. 你现在唯一的任务，是判断：在这个既定判断结果下，**说话人** 对 hypothesis 的倾向强度属于哪一个 confidence tier。
4. 这里的 confidence tier 表示 **说话人** 对 hypothesis 的倾向强弱，而不是模型对自己答案的自信度。
5. 你只能从以下四个 confidence tier 中选择一个：
- 弱
- 较弱
- 较强
- 强

当前 silver truth：{silver_truth}

请根据以下材料进行判断：
- 原始 text
- hypothesis
- 第一轮抽取结果
- 第二轮的 silver truth

再次强调：
- 不要重新判断 factivity
- 不要输出 TRUE、FALSE、UNCERTAIN
- 不要输出“正叙实”“反叙实”“非叙实”
- 只输出一个 confidence tier：弱 / 较弱 / 较强 / 强
- 这里的 confidence tier 指的是 **说话人** 对 hypothesis 的倾向强弱等级，是在既定 silver truth 下对 **说话人** 立场强度的进一步分档，不是模型对自己答案的自信度

输出格式：
请严格按照以下格式输出，不要输出其他内容：
<think>简短分析</think>
<confidence_tier>弱/较弱/较强/强</confidence_tier>

原始 text: {text}
hypothesis: {hypothesis}

第一轮抽取结果：
subject_type: {extraction["subject_type"]}
proposition_subject: {extraction["proposition_subject"]}
attitude_predicate: {extraction["attitude_predicate"]}
predicate_type: {extraction["predicate_type"]}
attitude_hint: {extraction["attitude_hint"]}
basis: {extraction["basis"]}

第二轮结果：
silver_truth: {silver_truth}"""


def _is_third_party_subject(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized == "third_party" or stripped == "第三方" or "第三方" in stripped


def _is_speaker_or_none_subject(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized in {"speaker", "none"} or stripped in {"说话人", "无"}


def _is_empty_basis(value: str) -> bool:
    stripped = value.strip()
    normalized = stripped.lower()
    return normalized in {"", "none"} or stripped == "无"


def build_expert_guidance(extraction: dict[str, str], prompt_lang: str) -> dict[str, str]:
    subject_type = extraction["subject_type"]
    basis = extraction["basis"]
    predicate_type = extraction.get("predicate_type", "非叙实")
    basis_is_empty = _is_empty_basis(basis)
    is_third_party = _is_third_party_subject(subject_type)

    if prompt_lang == "en":
        if not is_third_party:
            return {
                "expert_rule": "SPEAKER_MAIN_VIEW",
                "expert_advice": (
                    "Expert advice: The main viewpoint of the current sentence should be treated as the "
                    "**speaker**'s viewpoint. Judge the tendency of the **speaker** toward the hypothesis "
                    "directly from the whole sentence. Do not rewrite the task into judging some other "
                    "entity's tendency.\n\n"
                    f'Now integrate predicate_type="{predicate_type}" into your judgment: '
                    "if attitude_predicate is positive factive, it usually supports the hypothesis; "
                    "if it is negative factive, it usually rejects, refutes, or weakens the hypothesis; "
                    "if it is non-factive, it mainly expresses the **speaker**'s cognition, speculation, "
                    "emotion, desire, or attitude, and may not itself settle truth. However, if the whole "
                    "sentence already shows a directional tendency from the **speaker**, you should still "
                    "judge according to that tendency rather than mechanically outputting UNCERTAIN."
                ),
            }
        if basis_is_empty:
            return {
                "expert_rule": "THIRD_PARTY_NO_BASIS",
                "expert_advice": (
                    "Expert advice: The relevant subject is not the **speaker**, and there is **no basis** "
                    "in the sentence. Do not directly convert the third party's stance into the "
                    "**speaker**'s stance.\n\n"
                    f'Here predicate_type="{predicate_type}" is especially important. '
                    "If attitude_predicate is non-factive, it often first describes the third party's "
                    "cognition, speculation, emotion, desire, or attitude. In that case, further ask "
                    "whether the **speaker** is merely neutrally reporting that subjective state. Only if "
                    "it is truly neutral reporting and the **speaker** adds no extra directional signal "
                    "should you consider UNCERTAIN.\n\n"
                    "If attitude_predicate is positive factive or negative factive, then the predicate "
                    "itself more often means that the **speaker** has already added support, rejection, "
                    "correction, refutation, or weakening while reporting the third party's stance. In "
                    "that case, judge the **speaker**'s tendency from that added direction rather than "
                    "stopping at the third party's own stance."
                ),
            }
        return {
            "expert_rule": "THIRD_PARTY_WITH_BASIS",
            "expert_advice": (
                "Expert advice: The relevant subject is not the **speaker**, but there **is basis** in the "
                "sentence. This usually means the **speaker** is not merely neutral reporting the third "
                "party's stance, but is introducing information that supports some direction.\n\n"
                f'You should interpret basis together with predicate_type="{predicate_type}". '
                "If attitude_predicate is positive factive, the predicate and basis often jointly support "
                "the hypothesis. If it is negative factive, they often jointly support rejecting, "
                "refuting, or weakening the hypothesis. If it is non-factive, do not rely on the predicate "
                "alone, because non-factive predicates may not directly settle truth. Instead, ask whether "
                "the **speaker** uses the basis to turn what looks like a third party's attitude into the "
                "**speaker**'s own implicit tendency.\n\n"
                "So do not stop at the third party's stance. Combine basis, attitude_predicate, "
                "predicate_type, attitude_hint, and the whole sentence to judge the **speaker**'s "
                "tendency toward the hypothesis."
            ),
        }

    if not is_third_party:
        return {
            "expert_rule": "SPEAKER_MAIN_VIEW",
            "expert_advice": (
                "专家建议：当前句子的主视角按 **说话人** 处理。请直接根据整句话中 **说话人** 对 "
                "hypothesis 的倾向判断，不要改成判断其他主体的倾向。\n\n"
                f"同时要结合谓词类型理解该倾向的性质：当前 predicate_type 为“{predicate_type}”。"
                "如果 attitude_predicate 属于正叙实，通常说明该表达倾向于支持 hypothesis 成立；"
                "如果属于反叙实，通常说明该表达倾向于否定、反驳或削弱 hypothesis；"
                "如果属于非叙实，则说明该表达主要体现 **说话人** 的认知、推测、情绪、愿望或态度，"
                "本身未必直接决定命题真值，但只要整句话已经体现出 **说话人** 的方向性倾向，仍应据此判断，"
                "而不是机械地判为 UNCERTAIN。"
            ),
        }
    if basis_is_empty:
        return {
            "expert_rule": "THIRD_PARTY_NO_BASIS",
            "expert_advice": (
                "专家建议：当前相关主语不是 **说话人**，且句中 **无 basis**。请不要直接把第三方态度当成 "
                "**说话人** 的态度。\n\n"
                f"在这种情况下，谓词类型尤其重要：当前 predicate_type 为“{predicate_type}”。"
                "如果 attitude_predicate 属于非叙实，通常先表示第三方的认知、推测、情绪、愿望或态度；"
                "这时要进一步判断，**说话人** 是否只是中性地转述第三方的主观状态。只有在确实只是中性转述，"
                "且 **说话人** 没有额外带出方向性信息时，才考虑 UNCERTAIN。\n\n"
                "如果 attitude_predicate 属于正叙实或反叙实，则往往说明 **说话人** 在转述第三方态度时，"
                "已经通过该表达本身额外带出了支持、否定、纠正、反驳或削弱等方向，此时应优先判断 **说话人** 对 "
                "hypothesis 的倾向，而不是停留在第三方自己的态度上。"
            ),
        }
    return {
        "expert_rule": "THIRD_PARTY_WITH_BASIS",
        "expert_advice": (
            "专家建议：当前相关主语不是 **说话人**，且句中 **有 basis**。这通常说明 **说话人** 并非完全中性地"
            "转述第三方态度，而是在引入可支持某一方向的信息。\n\n"
            f"此时需要把 basis 和谓词类型结合起来理解：当前 predicate_type 为“{predicate_type}”。"
            "如果 attitude_predicate 属于正叙实，则该谓词与 basis 往往共同支持 hypothesis 的成立方向；"
            "如果属于反叙实，则该谓词与 basis 往往共同支持对 hypothesis 的否定、反驳或削弱；"
            "如果属于非叙实，则不能只看谓词本身，因为它未必直接决定命题真值，但在已有 basis 的情况下，"
            "应进一步判断 **说话人** 是否借由这些根据，把原本只是第三方态度的内容转化为自己对 hypothesis 的"
            "隐含倾向。\n\n"
            "因此，这一类样本不能只停留在第三方的态度上，而要结合 basis、attitude_predicate、"
            "predicate_type、attitude_hint 和整体语义，一起判断 **说话人** 对 hypothesis 的倾向。"
        ),
    }

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
    if is_complete_result(initial_predictions, data):
        print(
            "Existing result file is already complete. Recomputing accuracy and summaries without rerunning inference.",
            flush=True,
        )
        summary = save_main_results(
            result_path=result_path,
            dataset_path=args.dataset,
            provider=args.provider,
            model=model_name,
            prompt_lang=args.prompt_lang,
            predictions=initial_predictions,
        )
        save_error_results(
            errors_path=errors_path,
            dataset_path=args.dataset,
            provider=args.provider,
            model=model_name,
            prompt_lang=args.prompt_lang,
            predictions=initial_predictions,
            summary=summary,
        )
        print_summary(summary, result_path, errors_path)
        return

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
