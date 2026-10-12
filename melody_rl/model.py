"""Small Hugging Face / PEFT backend: MPS, CPU, or a single CUDA device."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Policy:
    def __init__(self, config, *, sft_adapter=None, rl_adapter=None, training=None):
        self.config = config
        self.device = choose_device(config["device"])
        seed_all(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["revision"], dtype=getattr(torch, config["dtype"]),
            attn_implementation="sdpa",
        ).to(self.device)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if sft_adapter:
            self.model = PeftModel.from_pretrained(self.model, str(sft_adapter)).merge_and_unload()
        if rl_adapter:
            self.model = PeftModel.from_pretrained(self.model, str(rl_adapter))
        elif training:
            # For RL the frozen base is the merged SFT policy. disable_adapter() is
            # an exact SFT reference, not the original language model.
            self.model = get_peft_model(self.model, LoraConfig(
                r=config["rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
                target_modules=config["target_modules"], task_type="CAUSAL_LM",
            ))
        if training and config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()

    def prompt_ids(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

    def generate(self, prompt, *, sample=False):
        return self.generate_many(prompt, count=1, sample=sample)[0]

    def generate_many(self, prompt, *, count, sample=True):
        """Batch completions of the same prompt; trim padding after each first EOS."""
        if count < 1 or self.config.get("rollout_batch_size", 1) < 1:
            raise ValueError("Generation count and rollout_batch_size must be positive")
        self.model.eval()
        inputs = self.prompt_ids(prompt)
        batch_size = min(count, self.config.get("rollout_batch_size", 1))
        results = []
        while len(results) < count:
            size = min(batch_size, count - len(results))
            results.extend(self._generate_batch(inputs.repeat(size, 1), sample=sample))
        return results

    def _generate_batch(self, inputs, *, sample):
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=self.config["max_new_tokens"], do_sample=sample,
                temperature=1.0, top_p=1.0, top_k=0 if sample else None,
                repetition_penalty=1.0, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [self.tokenizer.eos_token_id])
        # CPU conversion synchronizes device work before timing the complete batch.
        generated = output[:, inputs.shape[1]:].tolist()
        elapsed = time.perf_counter() - start
        results = []
        for index, tokens in enumerate(generated):
            stop = next((i + 1 for i, token in enumerate(tokens) if token in eos), None)
            tokens = tokens[:stop] if stop is not None else tokens
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            results.append({"text": text, "ids": output[index:index + 1, :inputs.shape[1] + len(tokens)].detach(),
                            "prompt_length": inputs.shape[1], "truncated": stop is None,
                            "tokens": len(tokens), "seconds": elapsed / len(generated)})
        return results

    def logps(self, ids, prompt_length):
        # Completion only, including EOS. No prompt/padding tokens in the objective.
        output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = output.logits[:, prompt_length - 1:-1, :].float()
        targets = ids[:, prompt_length:]
        return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)

    def target_ids(self, prompt, target):
        prefix = self.prompt_ids(prompt)
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        suffix = torch.tensor([target_ids], device=self.device)
        return torch.cat((prefix, suffix), dim=1), prefix.shape[1]

    def save(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory, save_embedding_layers=False)
        self.tokenizer.save_pretrained(directory)
        (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def clear_cache(self):
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
