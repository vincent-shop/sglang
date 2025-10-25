# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import time
from statistics import mean

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.speculative.cpp_ngram.ngram_cache import NgramCache


class TimeCollector:
    US = "us"

    def __init__(self):
        self.samples = []

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = (time.perf_counter() - self._start) * 1e6
        self.samples.append(duration)

    def dump_avg_max(self):
        if not self.samples:
            return [0.0, 0.0]
        return [mean(self.samples), max(self.samples)]


def prepare_contexts(args):
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    total_needed = args.num_token + args.branch_length
    prompts = [f"{args.prompt_text} {i}" for i in range(args.num_req)]
    contexts = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=total_needed,
                do_sample=args.sample,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            seq = generated[0].to("cpu").tolist()
            if len(seq) < total_needed:
                pad_token = tokenizer.pad_token_id or 0
                seq = [pad_token] * (total_needed - len(seq)) + seq
            contexts.append(seq[-total_needed:])
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return contexts


def main(args):
    rows = []
    history = prepare_contexts(args)
    for max_ngram in args.max_ngram:
        if max_ngram < args.min_ngram:
            continue

        cache = NgramCache(
            branch_length=args.branch_length,
            min_match_window_size=args.min_ngram,
            max_match_window_size=max_ngram,
            min_bfs_breadth=args.min_bfs_breadth,
            max_bfs_breadth=args.max_bfs_breadth,
            capacity=args.capacity,
            draft_token_num=args.num_spec_token,
        )
        cache.batch_put(history)
        cache.synchronize()

        queries = [row[-max_ngram:] for row in history]
        for _ in range(args.warmup):
            cache.batch_get(queries)
        gc.collect()

        collector = TimeCollector()
        for _ in range(args.num_iteration):
            with collector:
                cache.batch_get(queries)
        rows.append(
            [
                args.num_req,
                args.num_token,
                args.min_ngram,
                max_ngram,
                *collector.dump_avg_max(),
            ]
        )

    print(
        tabulate(
            rows,
            headers=["# Request", "# Token", "Min Ngram", "Max Ngram", "Avg (us)", "Max (us)"],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def invoke_main():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance of the SGLang n-gram proposer"
    )
    parser.add_argument("--num-iteration", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-req", type=int, default=64)
    parser.add_argument("--num-token", type=int, default=512)
    parser.add_argument("--min-ngram", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, nargs="*", default=[5, 7, 10, 15])
    parser.add_argument("--num-spec-token", type=int, default=4)
    parser.add_argument("--branch-length", type=int, default=18)
    parser.add_argument("--min-bfs-breadth", type=int, default=1)
    parser.add_argument("--max-bfs-breadth", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt-text", type=str, default="Benchmark prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()
