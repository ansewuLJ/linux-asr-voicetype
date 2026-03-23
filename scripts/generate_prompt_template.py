#!/usr/bin/env python3
"""
一次性工具：生成 prompt_template.json 与 mel_filters.npy。

来源：
- 上游项目： https://github.com/dseditor/QwenASRMiniTool.git
- 上游文件： generate_prompt_template.py

说明：
- 保持与上游核心生成逻辑一致。
- 仅增加参数化路径（--model-dir / --out-dir）以适配本仓库。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoModel, AutoProcessor


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate prompt_template.json and mel_filters.npy")
    ap.add_argument("--model-dir", type=Path, required=True, help="OpenVINO model directory")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory; default is model-dir")
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    out_dir = (args.out_dir or model_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("載入 qwen_asr processor…")
    import qwen_asr  # noqa
    from qwen_asr.inference.qwen3_asr import (  # type: ignore
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
        SUPPORTED_LANGUAGES,
    )

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(model_dir), fix_mistral_regex=True)

    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt_text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print(f"Prompt text: {repr(prompt_text)}")

    # 与上游一致：使用 480000 静音样本提取模板
    dummy_audio = np.zeros(480000, dtype=np.float32)
    inp = processor(text=[prompt_text], audio=[dummy_audio], return_tensors="np", padding=True)
    ids = inp["input_ids"][0].tolist()

    audio_pad_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    print(f"audio_pad_id  = {audio_pad_id}")
    print(f"Total tokens  = {len(ids)}")

    pad_positions = [i for i, x in enumerate(ids) if x == audio_pad_id]
    assert pad_positions, "找不到 <|audio_pad|>，請確認模型正確"
    print(f"Audio pad 位置：{pad_positions[0]}..{pad_positions[-1]}，共 {len(pad_positions)} 個")

    prefix_ids = ids[: pad_positions[0]]
    suffix_ids = ids[pad_positions[-1] + 1 :]
    print(f"Prefix IDs ({len(prefix_ids)}): {prefix_ids}")
    print(f"Suffix IDs ({len(suffix_ids)}): {suffix_ids}")

    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot_id = processor.tokenizer.eos_token_id or 151643

    special_ids = set()
    for tok_id_str, info in processor.tokenizer.added_tokens_decoder.items():
        if info.special:
            special_ids.add(int(tok_id_str))

    print(f"EOS id        = {eos_id}")
    print(f"EOT id        = {eot_id}")
    print(f"Special token count: {len(special_ids)}")

    asr_text_id = processor.tokenizer.convert_tokens_to_ids("<asr_text>")
    print(f"asr_text_id   = {asr_text_id}")

    language_suffix_ids: dict[str, list[int]] = {}
    for lang in SUPPORTED_LANGUAGES:
        lang_ids = processor.tokenizer.encode(f"language {lang}", add_special_tokens=False)
        language_suffix_ids[lang] = lang_ids + [asr_text_id]
    print(f"語系數量：{len(language_suffix_ids)}")

    template = {
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "n_audio_tokens": len(pad_positions),
        "audio_pad_id": audio_pad_id,
        "eos_id": eos_id,
        "eot_id": eot_id,
        "special_ids": sorted(special_ids),
        "prompt_text": prompt_text,
        "asr_text_id": asr_text_id,
        "language_suffix_ids": language_suffix_ids,
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }

    out_prompt = out_dir / "prompt_template.json"
    with open(out_prompt, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    fe = processor.feature_extractor
    mel_filters = fe.mel_filters
    out_mel = out_dir / "mel_filters.npy"
    np.save(str(out_mel), mel_filters)
    print(f"mel_filters shape: {mel_filters.shape} → {out_mel}")

    print(f"\n✅  已儲存至 {out_prompt}")
    print(f"✅  mel_filters 儲存至 {out_mel}")
    print(f"    prefix={len(prefix_ids)} tokens, audio_pad={len(pad_positions)}, suffix={len(suffix_ids)} tokens")
    print(f"    語系 suffix IDs 已預計算（{len(language_suffix_ids)} 種語系）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
