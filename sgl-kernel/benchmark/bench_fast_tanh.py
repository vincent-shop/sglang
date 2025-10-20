import argparse
import os
from typing import List

import sgl_kernel
import torch
import triton
import triton.testing

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def str2int_list(arg: str) -> List[int]:
    import re

    if arg in ("", None):
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]


if IS_CI:
    default_batch_sizes = [1]
    default_seq_lens = [1]
    default_dims = [1024]
else:
    default_batch_sizes = [1, 4, 16]
    default_seq_lens = [1, 64, 512]
    default_dims = [1024, 2048, 4096, 8192]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "dim"],
        x_vals=[],
        line_arg="dtype",
        line_vals=[torch.float16, torch.bfloat16],
        line_names=["FP16", "BF16"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="µs (median)",
        plot_name="gelu-tanh-fast-tanh-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, dim, dtype):
    device = torch.device("cuda")
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device=device)

    def run():
        return sgl_kernel.gelu_tanh_and_mul(x)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    ms, qmin, qmax = triton.testing.do_bench_cudagraph(run, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * qmax, 1000 * qmin


if __name__ == "__main__":
    p = argparse.ArgumentParser("fast_tanh benchmark")
    p.add_argument("--batch_sizes", type=str2int_list, default=default_batch_sizes)
    p.add_argument("--seq_lens", type=str2int_list, default=default_seq_lens)
    p.add_argument("--dims", type=str2int_list, default=default_dims)
    args = p.parse_args()

    if isinstance(args.batch_sizes, str):
        args.batch_sizes = str2int_list(args.batch_sizes)
    if isinstance(args.seq_lens, str):
        args.seq_lens = str2int_list(args.seq_lens)
    if isinstance(args.dims, str):
        args.dims = str2int_list(args.dims)

    import itertools

    benchmark_grid = list(
        itertools.product(args.batch_sizes, args.seq_lens, args.dims)
    )
    if hasattr(benchmark, "benchmarks"):
        benchmark.benchmarks.x_vals = benchmark_grid
    else:
        benchmark.benchmark.x_vals = benchmark_grid

    benchmark.run(print_data=True)
