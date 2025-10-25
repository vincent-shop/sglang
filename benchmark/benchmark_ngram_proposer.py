# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import time
from statistics import mean

import numpy as np
from tabulate import tabulate

from sglang.srt.speculative.cpp_ngram.ngram_cache import NgramCache


def benchmark_once(cache, queries):
    start = time.perf_counter()
    cache.batch_get(queries)
    end = time.perf_counter()
    return (end - start) * 1e6


def main(args):
    rows = []
    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(
        0,
        args.vocab_size,
        size=(args.num_req, args.num_token + args.branch_length),
        dtype=np.int32,
    )
    history = [row.tolist() for row in tokens]
    for max_window in args.max_window:
        cache = NgramCache(
            branch_length=args.branch_length,
            min_match_window_size=args.min_ngram,
            max_match_window_size=max_window,
            min_bfs_breadth=args.min_bfs_breadth,
            max_bfs_breadth=args.max_bfs_breadth,
            capacity=args.capacity,
            draft_token_num=args.num_spec_token,
        )
        cache.batch_put(history)
        cache.synchronize()
        queries = [row[-max_window:].tolist() for row in history]
        for _ in range(args.warmup):
            cache.batch_get(queries)
        gc.collect()
        measurements = [benchmark_once(cache, queries) for _ in range(args.num_iteration)]
        rows.append(
            [
                args.num_req,
                args.num_token,
                args.min_ngram,
                max_window,
                mean(measurements),
                max(measurements),
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
    parser.add_argument("--num-req", type=int, default=128)
    parser.add_argument("--num-token", type=int, default=2048)
    parser.add_argument("--min-ngram", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, nargs="*", default=[5, 7, 10, 15, 20])
    parser.add_argument("--num-spec-token", type=int, default=4)
    parser.add_argument("--branch-length", type=int, default=18)
    parser.add_argument("--min-bfs-breadth", type=int, default=1)
    parser.add_argument("--max-bfs-breadth", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()
