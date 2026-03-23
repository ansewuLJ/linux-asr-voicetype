# OpenVINO Runtime Assets (Third-Party Derived)

This directory contains runtime files required by `linux-asr-voicetype` OpenVINO backend:

- `prompt_template.json`
- `mel_filters.npy`

## Why these files are required

`src/voicetype/inference/openvino_backend.py` requires these files under model dir:

- `prompt_template.json`: precomputed prompt token template
- `mel_filters.npy`: precomputed mel filter matrix used by numpy processor

If they are missing, OpenVINO service startup will fail.

## Source / Attribution

These two files are copied from local project:

- `/home/lijie/code/QwenASRMiniTool/prompt_template.json`
- `/home/lijie/code/QwenASRMiniTool/ov_models/mel_filters.npy`

Reference project: `QwenASRMiniTool`

## License / Compliance note

- These files are third-party derived assets, not originally authored in this repository.
- Before redistribution/public release, verify upstream license terms of:
  - `QwenASRMiniTool` project
  - corresponding model artifacts and dependencies
- Keep this attribution file with the assets when distributing.

## Alternative (self-generate)

You may regenerate these files yourself from original model toolchain (for example via upstream generator script) to avoid directly reusing third-party generated artifacts.
