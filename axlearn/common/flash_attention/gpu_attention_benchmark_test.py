# Copyright © 2023 Apple Inc.

"""FlashAttention kernel benchmarks.


In addition to the dependencies in attention.py, also requires:
torch==2.1.0.dev20230726+cu121
pytorch-triton==2.1.0+9e3e10c5ed
"""
# pylint: skip-file

import jax
import jax.numpy as jnp
import triton  # pytype: disable=import-error

from axlearn.common.flash_attention.gpu_attention import flash_attention
from axlearn.common.flash_attention.utils import mha_reference
from jax.experimental.pallas.ops.attention import mha as pallas_mha

def _perf_report(prefix: str):
    batch_size, num_heads, seq_len, per_head_dim = 2, 40, 2048, 128

    # Vary num heads for fixed batch and seq length.
    num_heads_bench = triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[32, 40, 56], # Representing 3B, 13B and 30B models. Larger heads will be sharded.
        line_arg="library",
        line_vals=["jax", "jax-triton","jax-pallas"], 
        line_names=["Jax", "Jax Triton", "Pallas"],
        styles=[("blue", "-"), ("purple", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-batch{batch_size}-seq{seq_len}-d{per_head_dim}",
        args={"batch_size": batch_size, "seq_len": seq_len, "per_head_dim": per_head_dim},
    )
    # Vary seq length for fixed heads and batch size.
    seq_len_bench = triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(9, 13)],  # 512 to 4096.
        line_arg="library",
        line_vals=["jax", "jax-triton", "jax-pallas"],
        line_names=["Jax", "Jax Triton", "Pallas"],
        styles=[("blue", "-"), ("purple", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-batch{batch_size}-head{num_heads}-d{per_head_dim}",
        args={"batch_size": batch_size, "num_heads": num_heads, "per_head_dim": per_head_dim},
    )
    return triton.testing.perf_report(
        [num_heads_bench, seq_len_bench]
    )


@_perf_report("fwd")
def bench_flash_attention(
    batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, library: str
):
    warmup = 25
    rep = 10

    if library.startswith("jax"):
        q = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        k = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        v = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        bias = None
        #jax.random.normal(
         #   jax.random.PRNGKey(2), (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16
        #)

        if "triton" in library:
            fn = lambda: flash_attention(q, k, v, bias)
        elif "pallas" in library:
            fn = lambda: pallas_mha(q, k, v, segment_ids=None)
        else:
            fn = lambda: mha_reference(q, k, v, bias)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        
        
    else:
        raise ValueError(f"Unsupported: {library}")
    return ms


@_perf_report("grad")
def bench_flash_attention_backward(
    batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, library: str
):
    warmup = 25
    rep = 10

    if library.startswith("jax"):
        q = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        k = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        v = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        bias = None
        #  bias = jax.random.normal(
        #     jax.random.PRNGKey(3), (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16
        # )

        if "triton" in library:
            @jax.jit
            def test_fn(q, k, v, bias):
                return flash_attention(q, k, v, bias).sum()

            test_bwd = jax.grad(test_fn, argnums=(0, 1, 2))
            fn = lambda: test_bwd(q, k, v, bias)
        elif "pallas" in library:
            @jax.jit
            # No bias is supported yet.
            def pallas_fn(q, k, v):
                return pallas_mha(q, k, v, segment_ids=None).sum()

            pallas_bwd = jax.grad(pallas_fn, argnums=(0, 1, 2))
            fn = lambda: pallas_bwd(q, k, v)
        else:

            @jax.jit
            def ref_fn(q, k, v, bias):
                return mha_reference(q, k, v, bias).sum()

            ref_bwd = jax.grad(ref_fn, argnums=(0, 1, 2))
            fn = lambda: ref_bwd(q, k, v, bias)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    else:
        raise ValueError(f"Unsupported: {library}")
    return ms


bench_flash_attention.run(save_path=".", print_data=True)
bench_flash_attention_backward.run(save_path=".", print_data=True)
