# simulator.py

import time
from collections import defaultdict
from policies import LRUPolicy, LFUWithDecayPolicy, TinyLFUSLRUPolicy
from workload import generate_workload

class CacheSimulator:
    def __init__(self, cache_size, policy, n_blocks=100):
        self.cache_size = cache_size
        self.policy     = policy
        self.cache      = set()
        self.hits       = 0
        self.misses     = 0
        self.n_blocks   = n_blocks

    def access_block(self, block_id):
        blk = int(block_id)

        # 1) Normal hit/miss + update replacement policy
        if blk in self.cache:
            self.hits += 1
            self.policy.update(blk, hit=True)
        else:
            self.misses += 1
            if len(self.cache) >= self.cache_size:
                victim = self.policy.evict(self.cache)
                self.cache.remove(victim)
            self.cache.add(blk)
            self.policy.update(blk, hit=False)

        # 2) Next-line prefetch
        pre = (blk % self.n_blocks) + 1
        if pre not in self.cache:
            if len(self.cache) >= self.cache_size:
                victim = self.policy.evict(self.cache)
                self.cache.remove(victim)
            self.cache.add(pre)
            self.policy.update(pre, hit=False)

    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

def test_policy(factory, workload, cache_size, n_blocks=100):
    policy = factory(cache_size)
    sim = CacheSimulator(cache_size, policy, n_blocks=n_blocks)
    start = time.time()
    for blk in workload:
        sim.access_block(blk)
    return sim.get_hit_rate(), time.time() - start

if __name__ == "__main__":
    cache_sizes    = [16]  # only test k=16 for now you can add more sizes
    workload_types = ["uniform", "zipf", "cyclic"]
    seq_len        = 1_000_000
    n_blocks       = 100

    # 1) Test on synthetic workloads
    for wtype in workload_types:
        for cache_size in cache_sizes:
            print(f"\n=== Testing k={cache_size} on {wtype} distribution ===")
            wl = generate_workload(n_blocks=n_blocks,
                                   length=100_000,
                                   wtype=wtype)

            tests = [
                ("LRU (baseline)",         lambda k: LRUPolicy()),
                ("LFU-Decay (custom)",     lambda k: LFUWithDecayPolicy()),
                ("TinyLFU+SLRU (advanced)",lambda k: TinyLFUSLRUPolicy(cache_size=k)),
            ]

            for name, factory in tests:
                hr, t = test_policy(factory, wl, cache_size=cache_size, n_blocks=n_blocks)
                print(f"  {name:25s} → Hit Rate: {hr:.2%}, Time: {t:.2f}s")

    # 2) Hidden workload or fallback
    cache_size = 16
    try:
        from workload import load_hidden_workload
        wl = load_hidden_workload(length=seq_len)
        print("\nLoaded hidden workload successfully")
    except Exception:
        print("\nHidden workload not available, generating synthetic mixed workload")
        zipf_part    = generate_workload(n_blocks=80, length=400_000, wtype="zipf")
        cyclic_part  = generate_workload(n_blocks=30, length=300_000, wtype="cyclic")
        uniform_part = generate_workload(n_blocks=100, length=300_000, wtype="uniform")
        wl = list(zipf_part) + list(cyclic_part) + list(uniform_part)

    print(f"\n=== Testing k={cache_size} on hidden distribution ===")
    tests = [
        ("LRU (baseline)",         lambda k: LRUPolicy()),
        ("LFU-Decay (custom)",     lambda k: LFUWithDecayPolicy()),
        ("TinyLFU+SLRU (advanced)",lambda k: TinyLFUSLRUPolicy(cache_size=k)),
    ]

    for name, factory in tests:
        hr, t = test_policy(factory, wl, cache_size=cache_size, n_blocks=n_blocks)
        print(f"  {name:25s} → Hit Rate: {hr:.2%}, Time: {t:.2f}s")
