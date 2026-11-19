"""Chạy một checkpoint DeepSeek-R1-Distill-Qwen trên một GPU CUDA."""

import argparse
import copy
import gc
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
import yaml
from filelock import FileLock
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .dataset import load_dataset, normalize_dataset, validate_dataset
from .inference_checkpoint import batch_is_complete, finalize_results, save_batch, save_manifest


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} phải là số nguyên dương.")
    return value


def _load_questions(path):
    if "fixtures" in path.parts or path.name.casefold().startswith("synthetic"):
        raise ValueError("Hãy chọn dataset thật đã rà soát để chạy inference.")
    dataset = normalize_dataset(load_dataset(path))
    issues = validate_dataset(dataset)
    if not issues.empty:
        raise ValueError(f"Dataset có lỗi. Kiểm tra bằng notebook 01.\n{issues.head(10).to_string(index=False)}")
    return list(dataset[["id", "question"]].itertuples(index=False, name=None))


def _resolve_model(config, model_id):
    matches = [entry for entry in config["models"] if entry["id"] == model_id]
    if len(matches) != 1 or not matches[0]["enabled"]:
        raise ValueError("Chọn chính xác một model ID đã bật trong configs.")

    entry = matches[0]
    api = HfApi()
    model_ref = entry.get("model_revision") or "main"
    tokenizer_ref = entry.get("tokenizer_revision") or model_ref
    model_revision = api.model_info(model_id, revision=model_ref).sha
    tokenizer_revision = model_revision
    if tokenizer_ref != model_ref:
        tokenizer_revision = api.model_info(model_id, revision=tokenizer_ref).sha
    return {"id": model_id, "size": entry["size"],
            "model_revision": model_revision, "tokenizer_revision": tokenizer_revision}


def _load_model(spec, dtype_name, quantization):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    kwargs = {"dtype": dtype, "device_map": {"": 0}, "use_safetensors": True}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["tokenizer_revision"], padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["model_revision"], **kwargs
    ).eval()
    return model, tokenizer


def _render_prompt(tokenizer, template, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(question=question)}],
        tokenize=False, add_generation_prompt=True,
    )


def _generate_batch(model, tokenizer, prompts, generation, seed):
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
    prompt_width = inputs["input_ids"].shape[1]
    if prompt_width + generation.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt và ngân sách sinh vượt cửa sổ ngữ cảnh.")

    inputs = inputs.to("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.synchronize(0)
    start = perf_counter()
    with torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=generation)
    torch.cuda.synchronize(0)
    elapsed = perf_counter() - start

    eos_ids = generation.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    outputs = []
    for generated in sequences[:, prompt_width:].tolist():
        # Các hàng kết thúc sớm được pad sau EOS. Đếm EOS, nhưng bỏ nó khi giải mã.
        end = next((i for i, token in enumerate(generated) if token in eos_ids), None)
        text_ids = generated if end is None else generated[:end]
        outputs.append({
            "generated_response": tokenizer.decode(text_ids, clean_up_tokenization_spaces=False),
            "generated_token_count": len(generated) if end is None else end + 1,
            "stop_reason": "max_new_tokens" if end is None else "eos",
        })
    return outputs, elapsed


def _run_pending_batches(pending, directory, manifest):
    if not torch.cuda.is_available():
        raise RuntimeError("Cần phiên Colab GPU có PyTorch nhận được CUDA.")
    options = manifest["inference"]
    dtype = options["dtype"]
    compute_major, _ = torch.cuda.get_device_capability(0)
    if dtype == "bfloat16" and compute_major < 8:
        raise RuntimeError("GPU hiện tại không hỗ trợ bfloat16 phần cứng.")
    if options["quantization"] == "4bit" and compute_major < 6:
        raise RuntimeError("NF4 yêu cầu GPU NVIDIA Pascal trở lên.")

    packages = ["torch", "transformers", "accelerate", "huggingface_hub", "tokenizers"]
    if options["quantization"] == "4bit":
        packages.append("bitsandbytes")
    environment = {"packages": {name: version(name) for name in packages},
                   "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    spec = manifest["model"]
    model = None
    try:
        model, tokenizer = _load_model(spec, dtype, options["quantization"])
        generation = copy.deepcopy(model.generation_config)
        generation.update(**options["generation"], pad_token_id=tokenizer.pad_token_id,
                          num_beams=1, num_return_sequences=1, return_dict_in_generate=False)
        effective_config = json.loads(generation.to_json_string(use_diff=False))
        save_manifest(directory / "generation_config.json", effective_config)

        for path, batch, generation_index, seed in pending:
            prompts = [_render_prompt(tokenizer, manifest["prompt"]["user_template"], question)
                       for _, question in batch]
            outputs, elapsed = _generate_batch(model, tokenizer, prompts, generation, seed)
            rows = []
            for (question_id, _), prompt, output in zip(batch, prompts, outputs, strict=True):
                rows.append({
                    "question_id": question_id,
                    "generation_index": generation_index,
                    "model_name": spec["id"],
                    "model_size": spec["size"],
                    "prompt": prompt,
                    **output,
                    "inference_time": elapsed / len(batch),
                    "batch_time_seconds": elapsed,
                    "batch_size": len(batch),
                    "batch_seed": seed,
                    "generation_config": effective_config,
                    "environment": environment,
                })
            save_batch(path, rows)
            print(f"Đã lưu {path.name} ({len(rows)} bản ghi)", flush=True)
    except torch.cuda.OutOfMemoryError as error:
        # Traceback của batch có thể còn giữ tham chiếu đến tensor CUDA.
        error.__traceback__ = None
        raise RuntimeError(
            "GPU hết bộ nhớ. Checkpoint đã lưu vẫn được giữ. Tiếp tục trên GPU đủ VRAM, "
            "hoặc dùng run_id mới với batch_size nhỏ hơn hay lượng tử hóa."
        ) from None
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_inference(config_path, model_id, run_id, *, project_root=Path("."), output_dir=None):
    """Cùng run_id tiếp tục lần chạy cũ. Dùng run_id mới để chủ động chạy lại."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("run_id cần dài 1–80 ký tự và chỉ dùng chữ ASCII, số, gạch ngang hoặc gạch dưới.")
    root = Path(project_root).resolve()
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    dataset = config["dataset"]
    if not dataset.get("path"):
        raise ValueError("Điền dataset.path trong cấu hình.")
    data_path = (root / dataset["path"]).resolve()
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    questions = _load_questions(data_path)

    inference = config["inference"]
    prompt = config["prompt"]
    if prompt.get("system_message") is not None:
        raise ValueError("Giao thức zero-shot dùng duy nhất một user message.")
    if "{question}" not in prompt["user_template"]:
        raise ValueError("prompt.user_template phải có {question}.")
    batch_size = _positive_integer(inference["batch_size"], "batch_size")
    repeats = _positive_integer(inference["generations_per_question"], "generations_per_question")
    seeds = inference.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != repeats
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != repeats):
        raise ValueError("Điền một seed nguyên 32-bit không âm và khác nhau cho mỗi lần sinh.")
    options = {
        "dtype": inference.get("dtype") or "float16",
        "quantization": inference.get("quantization"),
        "batch_size": batch_size,
        "seeds": seeds,
        "generation": {
            "max_new_tokens": _positive_integer(inference.get("max_new_tokens"), "max_new_tokens"),
            "do_sample": inference["do_sample"],
            "temperature": inference["temperature"],
            "top_p": inference["top_p"],
        },
    }
    if options["dtype"] not in ("float16", "bfloat16") or options["quantization"] not in (None, "4bit"):
        raise ValueError("Dùng dtype float16/bfloat16 và quantization null/4bit.")
    directory = (root / (output_dir or config["output"]["directory"]) / run_id).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "predictions.jsonl"

    with FileLock(str(directory / ".run.lock"), timeout=0):
        manifest = {"format_version": 1, "run_id": run_id, "dataset_sha256": digest,
                    "dataset_version": dataset.get("version"), "question_count": len(questions),
                    "model": _resolve_model(config, model_id), "inference": options, "prompt": prompt}
        save_manifest(directory / "manifest.json", manifest)
        checkpoint_dir = directory / "batches"
        checkpoint_dir.mkdir(exist_ok=True)
        paths, pending = [], []
        for repeat, seed in enumerate(seeds, start=1):
            for index, start in enumerate(range(0, len(questions), batch_size)):
                batch = questions[start:start + batch_size]
                path = checkpoint_dir / f"r{repeat:03d}-b{index:06d}.jsonl"
                paths.append(path)
                keys = [(question_id, repeat) for question_id, _ in batch]
                if not batch_is_complete(path, keys):
                    # Phân luồng RNG theo lần sinh và batch, độc lập với thứ tự khôi phục.
                    pending.append((path, batch, repeat, (seed << 32) | index))
        if pending:
            _run_pending_batches(pending, directory, manifest)
        finalize_results(result_path, paths)
    return result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(run_inference(args.config, args.model_id, args.run_id, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
