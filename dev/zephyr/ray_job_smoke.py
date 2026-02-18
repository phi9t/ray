#!/usr/bin/env python3
"""Ray runtime smoke workload for Zephyr container validation."""

import time

import ray


@ray.remote
def count_primes(limit: int) -> int:
    count = 0
    for n in range(2, limit):
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            count += 1
    return count


@ray.remote
class Accumulator:
    def __init__(self):
        self.total = 0

    def add(self, value: int) -> int:
        self.total += value
        return self.total

    def get(self) -> int:
        return self.total


def main() -> None:
    start = time.time()
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=True)

    limits = [20000 + i * 250 for i in range(24)]
    counts = ray.get([count_primes.remote(limit) for limit in limits])

    actor = Accumulator.remote()
    ray.get([actor.add.remote(v) for v in counts])
    final_total = ray.get(actor.get.remote())

    payload_len = 32 * 1024 * 1024
    obj = ray.put(b"x" * payload_len)
    roundtrip_len = len(ray.get(obj))

    assert final_total == sum(counts)
    assert roundtrip_len == payload_len

    print("ray_version=", ray.__version__)
    print("nodes=", len(ray.nodes()))
    print("tasks=", len(counts))
    print("sum_counts=", final_total)
    print("object_roundtrip_bytes=", roundtrip_len)
    print("elapsed_sec=", round(time.time() - start, 3))

    ray.shutdown()


if __name__ == "__main__":
    main()
