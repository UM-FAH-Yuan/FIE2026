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
DEFAULT_OUTPUT_DIR = ROOT / "outputs_v2_revised"
VALID_LABELS = ("TRUE", "FALSE", "UNCERTAIN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate factivity labels on the FIE2026 sample set with a single-turn prompt."
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


# def build_single_turn_prompt(text: str, hypothesis: str, prompt_lang: str) -> str:
#     if prompt_lang == "en":
#         return f"""Task: Given the text and hypothesis, determine the factivity label of the hypothesis in the context of the text.

# This task is not asking how confident you are in your own answer. Instead, you must judge the stance of the **speaker** expressed in the text toward the proposition in the hypothesis.

# Please note:
# 1. You must judge whether, in the text, the subject treats the proposition in the hypothesis as true, false, or uncertain.
# 2. TRUE means: the text shows that the subject treats the proposition expressed by the hypothesis as true.
# 3. FALSE means: the text shows that the subject treats the proposition expressed by the hypothesis as false.
# 4. UNCERTAIN means: the text shows that the subject does not make a truth-value commitment about the proposition in the hypothesis, or only expresses uncertain attitudes such as guessing, worrying, hoping, or believing.
# 5. Do not interpret the task as asking whether you yourself are confident. You must identify the subject's stance in the text, not your own stance.

# Output format:
# You must strictly follow this format and output nothing else:
# <think>your analysis</think>
# <answer>TRUE</answer>

# The answer must be exactly one of: TRUE, FALSE, UNCERTAIN.

# text: {text}
# hypothesis: {hypothesis}"""

#     return f"""任务：根据给定的 text 和 hypothesis，判断 hypothesis 在 text 语境中的叙实性标签。

# 这里判断的不是“你这个模型对答案有多确定”，而是 text 中的**说话人**，对 hypothesis 所表达命题的态度是什么。

# 请注意：
# 1. 你要判断的是：在 text 中，主体是否把 hypothesis 对应的命题当作真、假，或不确定。
# 2. TRUE 表示：text 体现出主体将 hypothesis 对应的命题视为真。
# 3. FALSE 表示：text 体现出主体将 hypothesis 对应的命题视为假。
# 4. UNCERTAIN 表示：text 体现出主体对 hypothesis 对应的命题不作真值承诺，或仅表达猜测、担心、希望、认为等不确定态度。
# 5. 不要把任务理解成“你自己是否有把握”；这里要判断的是 text 中主体的立场，而不是你的立场。

# 输出要求：
# 请严格按照以下格式输出，不要输出其他格式：
# <think>你的分析</think>
# <answer>TRUE</answer>

# 其中 answer 只能是 TRUE、FALSE、UNCERTAIN 三者之一。

# text: {text}
# hypothesis: {hypothesis}"""

def build_single_turn_prompt(text: str, hypothesis: str, prompt_lang: str) -> str:
    if prompt_lang == "en":
        return f"""Task: Given the text and hypothesis, determine the factivity label of the hypothesis in the context of the text.

This task is not asking how confident you are in your own answer. You must judge the stance expressed in the text toward the proposition in the hypothesis.

Core principle:
You are judging whether the proposition in the hypothesis is presented in the text with a positive tendency, a negative tendency, or only uncertainty.

Important distinctions:
1. If the grammatical subject in the text is also the cognitive subject, then a tendency already matters.
   If the subject shows a positive tendency toward the proposition, output TRUE.
   If the subject shows a negative tendency toward the proposition, output FALSE.
   It does not need to be a fully certain assertion.
   Verbs or expressions like "feel", "estimate", "infer", "seem", or other weakly directional attitudes may still indicate TRUE or FALSE if they show a clear tendency.

2. If the speaker/narrator and the grammatical subject are different, Instead, examine what stance is attributed to the grammatical subject inside the sentence.

3. When the attributed stance is only a pure guess, belief, or other non-committal mental attitude with no clear factual or evidential grounding, output UNCERTAIN.

4. But if the text gives factual basis, perceptual basis, investigative basis, authority basis, documentary basis, or other evidential support, then do not default to UNCERTAIN.
   In such cases, even if the expression is "infer", "estimate", "suspect", "accuse", or similar, you should judge whether the overall tendency is positive or negative.

5. In short:
   - positive tendency toward the proposition -> TRUE
   - negative tendency toward the proposition -> FALSE
   - pure uncertainty without real commitment or grounding -> UNCERTAIN

Do not interpret the task as asking whether you yourself are confident. You must identify the stance encoded in the text.

Output format:
You must strictly follow this format and output nothing else:
<think>your analysis</think>
<answer>TRUE</answer>

The answer must be exactly one of: TRUE, FALSE, UNCERTAIN.

text: {text}
hypothesis: {hypothesis}"""

    return f"""任务：根据给定的 text 和 hypothesis，判断 hypothesis 在 text 语境中的叙实性标签。

这里判断的不是“你这个模型对答案有多确定”，而是 text 中体现出的、针对 hypothesis 所表达命题的立场。

核心原则：
你要判断的是，hypothesis 对应的命题在 text 中究竟呈现为正向倾向、反向倾向，还是仅仅是不确定。

请特别注意以下区分：

1. 如果句子的主语本身就是认知主体，那么“有倾向”就已经很重要。
   只要主语对该命题表现出正向倾向，就应判为 TRUE。
   只要主语对该命题表现出反向倾向，就应判为 FALSE。
   不要求一定是完全确定的断言。
   比如“感觉、估计、料想、似乎”等，只要对命题有明确偏向，也可能是 TRUE 或 FALSE，而不一定是 UNCERTAIN。

2. 如果说话人/叙述者和句子里的主语不是同一个主体，要看句子里归属于那个主语的立场是什么。

3. 如果该主语表达的只是纯猜测、纯认为、担心等，没有明显事实根据、证据根据或权威根据，只是非承诺性的心理态度，那么应判为 UNCERTAIN。

4. 但如果 text 中给出了事实根据、感知根据、调查根据、权威根据、文献根据或其他证据支持，就不要默认判为 UNCERTAIN。
   这时即使出现“推测、估算、怀疑、控告”等词，也要结合整句看它最终是正向倾向还是否向倾向。

5. 简单说：
   - 对命题有正向倾向 -> TRUE
   - 对命题有反向倾向 -> FALSE
   - 只有不确定、没有真实承诺或缺乏根据 -> UNCERTAIN

不要把任务理解成“你自己是否有把握”；这里要判断的是 text 中编码出来的立场，而不是你的立场。

输出要求：
请严格按照以下格式输出，不要输出其他格式：
<think>你的分析</think>
<answer>TRUE</answer>

其中 answer 只能是 TRUE、FALSE、UNCERTAIN 三者之一。

text: {text}
hypothesis: {hypothesis}"""

# V3
# def build_single_turn_prompt(text: str, hypothesis: str, prompt_lang: str) -> str:
#     if prompt_lang == "en":
#         return f"""Task: Given the text and hypothesis, determine the factivity label of the hypothesis in the context of the text.

# This task is not asking how confident you are in your own answer. You must identify the tendency expressed by the speaker in the text toward the proposition in the hypothesis.

# Core principle:
# What matters most is the speaker's tendency toward the proposition expressed by the hypothesis.
# You must judge whether the speaker presents that proposition with a positive tendency, a negative tendency, or an uncertain tendency.

# Important points:
# 1. Focus on the speaker's encoded stance in the sentence, not on your own certainty.
# 2. A full explicit assertion is not required. An implicit tendency also counts.
# 3. Pay close attention to lexical meaning, presupposition, connotation, evaluative polarity, semantic direction, and directional bias carried by words and constructions.
#    Expressions such as "realize", "know", "notice", "discover", "forget", "mistakenly", "falsely", "pretend", "boast", "worry", "guess", "dream", "seem", and similar items may carry implicit positive, negative, or uncertain orientation toward the proposition.
# 4. Distinguish carefully between pure uncertainty and evidentially grounded judgment.
#    If the sentence contains factual basis, perceptual basis, investigative basis, documentary basis, expert basis, authority basis, or other evidential grounding, do not reduce it to UNCERTAIN too quickly.
# 5. If the sentence only conveys pure guessing, pure hoping, pure worrying, pure imagining, pure dreaming, or other weak non-committal mentality, with no real commitment and no factual grounding, then it is more likely UNCERTAIN.
# 6. Even when the sentence is not a strong factual assertion, if the wording itself carries a clear positive or negative semantic tendency toward the proposition, that tendency should count.
# 7. When the sentence involves another person's belief, claim, or attitude, still determine what tendency the speaker ultimately encodes toward the proposition in the hypothesis.

# Decision rule:
# - positive tendency toward the proposition -> TRUE
# - negative tendency toward the proposition -> FALSE
# - uncertain tendency without real commitment -> UNCERTAIN

# Output format:
# You must strictly follow this format and output nothing else:
# <think>your analysis</think>
# <answer>TRUE</answer>

# The answer must be exactly one of: TRUE, FALSE, UNCERTAIN.

# text: {text}
# hypothesis: {hypothesis}"""

#     return f"""任务：根据给定的 text 和 hypothesis，判断 hypothesis 在 text 语境中的叙实性标签。

# 这里判断的不是“你这个模型对答案有多确定”，而是 text 中说话人对 hypothesis 所表达命题的倾向。

# 核心原则：
# 最重要的是判断说话人对该命题的倾向。
# 你要判断的是：说话人在这句话里，究竟把该命题表达为正向倾向、反向倾向，还是不确定倾向。

# 请特别注意：
# 1. 关注说话人在句中编码出来的立场，而不是你自己的把握。
# 2. 不要求一定出现明确、强硬、完整的断言。只要有隐含倾向，也算倾向。
# 3. 要特别注意词语和句式本身携带的隐含倾向，包括：
#    词义、预设、褒贬色彩、评价方向、语义偏向、构式触发的方向性。
#    例如“知道、意识到、注意到、发现、忘记、错误地、诬陷、假装、吹嘘、担心、猜测、梦想、似乎”等词语或结构，往往会对命题带来隐含的正向、反向或不确定倾向。
# 4. 要正确区分“纯不确定”与“有事实依据的判断”。
#    如果句中有事实根据、感知根据、调查根据、文献根据、专家根据、权威根据或其他证据支撑，就不要轻易判成 UNCERTAIN。
# 5. 如果只是纯猜测、纯希望、纯担心、纯幻想、纯梦想等，没有真实承诺，也缺少事实或证据支撑，才更可能是 UNCERTAIN。
# 6. 即使不是强事实陈述，只要措辞本身已经对命题带有明确的正向或反向语义倾向，就应把这种倾向计入判断。
# 7. 当句子里涉及他人的看法、想法、说法时，仍然要看整句话最终编码出来的、针对 hypothesis 命题的说话人倾向是什么。

# 判断规则：
# - 对命题呈现正向倾向 -> TRUE
# - 对命题呈现反向倾向 -> FALSE
# - 对命题呈现不确定倾向，且缺乏真实承诺 -> UNCERTAIN

# 输出要求：
# 请严格按照以下格式输出，不要输出其他格式：
# <think>你的分析</think>
# <answer>TRUE</answer>

# 其中 answer 只能是 TRUE、FALSE、UNCERTAIN 三者之一。

# text: {text}
# hypothesis: {hypothesis}"""



def normalize_label(value: str) -> str:
    label = value.strip().upper()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {value!r}")
    return label


def extract_answer_label(raw_text: str) -> str:
    raw_text = raw_text.strip()
    match = re.search(r"<answer>\s*(TRUE|FALSE|UNCERTAIN)\s*</answer>", raw_text, re.IGNORECASE)
    if match:
        return normalize_label(match.group(1))

    # Fallback for models that return only the bare label.
    bare = raw_text.strip()
    if bare.upper() in VALID_LABELS:
        return normalize_label(bare)

    raise ValueError(f"Could not extract label from model output: {raw_text!r}")


@dataclass
class Prediction:
    sample_id: str
    gold: str
    pred: str
    ok: bool
    raw_output: str


class MockSingleTurnClient:
    def predict(self, text: str, hypothesis: str) -> str:
        # This branch is a local smoke test only. It simulates the model I/O shape
        # with a small set of lexical rules so the evaluation pipeline can run offline.
        del hypothesis
        text_lower = text.lower()

        false_markers = (
            "错误地认为",
            "假装",
            "吹嘘",
            "诬陷",
            "幻想",
            "梦想",
            "嫁祸",
            "并没有计划",
            "没有试图",
            "不能认同",
            "不该相信",
        )
        uncertain_markers = (
            "认为",
            "猜测",
            "担心",
            "希望",
            "不知道",
            "觉得",
            "估计",
            "推测",
            "似乎",
            "料想",
        )

        label = "TRUE"
        if any(marker in text_lower for marker in false_markers):
            label = "FALSE"
        elif any(marker in text_lower for marker in uncertain_markers):
            label = "UNCERTAIN"

        return f"<think>mock smoke test</think><answer>{label}</answer>"


class OpenAICompatibleChatClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None,
        max_retries: int,
        prompt_lang: str,
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
        self.prompt_lang = prompt_lang
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**client_kwargs)

    def predict(self, text: str, hypothesis: str) -> str:
        prompt = build_single_turn_prompt(text, hypothesis, self.prompt_lang)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=1,
                    top_p=0.2,
                    max_tokens=512,
                )
                content = response.choices[0].message.content.strip()
                print('-'*50)
                print(content)
                print('-'*50)
                if content is None:
                    raise ValueError("Model returned empty content.")
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise RuntimeError(f"Chat completion request failed: {exc}") from exc

        raise RuntimeError(f"Prediction failed after retries: {last_error}")


def make_client(args: argparse.Namespace) -> Any:
    if args.provider == "mock":
        return MockSingleTurnClient()
    if not args.api_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY or pass --api-key, or use --provider mock."
        )
    return OpenAICompatibleChatClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
        prompt_lang=args.prompt_lang,
    )


def evaluate(
    data: list[dict[str, Any]],
    client: Any,
    sleep_seconds: float,
    verbose: bool,
) -> tuple[list[Prediction], dict[str, Any]]:
    predictions: list[Prediction] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for index, item in enumerate(data, start=1):
        raw_output = client.predict(item["text"], item["hypothesis"])
        pred_label = extract_answer_label(raw_output)
        gold_label = normalize_label(item["factivity"])
        ok = pred_label == gold_label

        predictions.append(
            Prediction(
                sample_id=item["id"],
                gold=gold_label,
                pred=pred_label,
                ok=ok,
                raw_output=raw_output,
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
    result_path = output_dir / f"factivity_label_eval_{provider}_{model}_ZH_T01_TOPP0_2.json"
    errors_path = output_dir / f"factivity_label_errors_{provider}_{model}_ZH_T01_TOPP0_2.json"
    payload = {
        "dataset": str(dataset_path),
        "provider": provider,
        "model": model,
        "prompt_lang": prompt_lang,
        "summary": summary,
        "predictions": [
            {
                "id": item.sample_id,
                "gold": item.gold,
                "pred": item.pred,
                "ok": item.ok,
                "raw_output": item.raw_output,
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
                "gold": item.gold,
                "pred": item.pred,
                "ok": item.ok,
                "raw_output": item.raw_output,
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
        print(
            f"{label}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})"
        )
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
