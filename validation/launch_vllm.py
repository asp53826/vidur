"""Print the `vllm serve` command for an experiment file.

Settings the simulator needs to mirror are pinned explicitly rather than left
to vLLM defaults, which change between releases:

  --max-num-batched-tokens  -> Vidur's Sarathi chunk_size (token budget)
  --max-num-seqs            -> Vidur's batch_size_cap
  --block-size              -> Vidur's block_size
  --no-enable-prefix-caching  Vidur doesn't model prefix caching, and v1
                              turns it on by default
"""

import json
import shlex
import sys


def command(experiment: dict, port: int) -> list:
    cmd = [
        "vllm",
        "serve",
        experiment["model"],
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(experiment.get("tensor_parallel_size", 1)),
        "--max-model-len",
        str(experiment["max_model_len"]),
        "--max-num-batched-tokens",
        str(experiment["max_num_batched_tokens"]),
        "--max-num-seqs",
        str(experiment["max_num_seqs"]),
        "--block-size",
        str(experiment.get("block_size", 16)),
        "--no-enable-prefix-caching",
        "--seed",
        "0",
    ]
    spec = experiment.get("speculative_config")
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    return cmd


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: launch_vllm.py EXPERIMENT_JSON [PORT]")
    with open(sys.argv[1]) as f:
        exp = json.load(f)
    port = int(sys.argv[2]) if len(sys.argv) == 3 else 8000
    print(shlex.join(command(exp, port)))
