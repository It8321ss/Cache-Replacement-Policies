# workload.py

import numpy as np

def generate_workload(n_blocks=100, length=1_000_000, wtype="uniform"):
    """
    Generate a sequence of block‑ID requests.
    wtype: one of "uniform", "zipf", "cyclic", "phased"
    """
    if wtype == "uniform":
        return np.random.randint(1, n_blocks+1, size=length)
    elif wtype == "zipf":
        return np.random.zipf(a=1.5, size=length) % n_blocks + 1
    elif wtype == "cyclic":
        # 1,2,3,…,n_blocks, 1,2,3… repeatedly
        return np.array([(i % n_blocks) + 1 for i in range(length)])
    elif wtype == "phased":
        phase_len = length // 4
        seq = []
        for p in range(4):
            start = p * (n_blocks // 4) + 1
            end   = (p+1) * (n_blocks // 4)
            seq.extend(
                np.random.randint(start, end+1, size=phase_len).tolist()
            )
        if len(seq) < length:
            seq.extend(np.random.randint(1, n_blocks+1, size=length-len(seq)).tolist())
        return np.array(seq)
    else:
        raise ValueError(f"Unknown workload type: {wtype}")
