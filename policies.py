import time
from collections import OrderedDict, defaultdict
import numpy as np

# ─── 1. Baseline: LRU ──────────────────────────────────────────────────────────
class LRUPolicy:
    def __init__(self):
        self.order = OrderedDict()

    def update(self, blk, hit=False):
        if blk in self.order:
            self.order.pop(blk)
        self.order[blk] = True

    def evict(self, cache):
        # evict the LRU block that's still in cache
        for key in self.order:
            if key in cache:
                self.order.pop(key)
                return key
        # fallback
        key, _ = self.order.popitem(last=False)
        return key


# ─── 2. Custom: LFU with Decay ────────────────────────────────────────────────
class LFUWithDecayPolicy:
    def __init__(self, decay_interval=10000, decay_factor=0.5):
        self.freqs = defaultdict(int)
        self.counter = 0
        self.decay_interval = decay_interval
        self.decay_factor = decay_factor

    def update(self, blk, hit=False):
        self.freqs[blk] += 1
        self.counter += 1
        if self.counter % self.decay_interval == 0:
            for k in list(self.freqs):
                self.freqs[k] *= self.decay_factor

    def evict(self, cache):
        # evict the block in cache with smallest freq
        victim = min(cache, key=lambda k: self.freqs.get(k, 0))
        self.freqs.pop(victim, None)
        return victim


from collections import OrderedDict, deque, defaultdict

class TinyLFUSLRUPolicy:
    """
    TinyLFU+SLRU with:
      - ghost list for probation evictions (A1out)
      - adjust probation_fraction up/down on ghost hits
    """

    def __init__(self,
                 cache_size,
                 init_prob_fraction=0.2,
                 freq_window=10000,
                 adjust_interval=50000,
                 adjust_step=0.05):
        self.c = cache_size

        # dynamic probation fraction
        self.prob_fraction = init_prob_fraction
        self._repartition()

        # segments
        self.prob      = deque()               # probation FIFO
        self.prot      = OrderedDict()         # protected LRU
        self.ghost_prob= deque(maxlen=self.prob_size)  # ghost for probation

        # TinyLFU filter
        self.freq      = defaultdict(int)
        self.window    = deque(maxlen=freq_window)

        # adaptivity counters
        self.accesses     = 0
        self.ghost_hits   = 0
        self.adjust_interval = adjust_interval
        self.adjust_step     = adjust_step

    def _repartition(self):
        # recalc sizes from current fraction
        self.prob_size = max(1, int(self.c * self.prob_fraction))
        self.prot_size = self.c - self.prob_size
        # resize ghost if needed
        if hasattr(self, 'ghost_prob'):
            self.ghost_prob = deque(self.ghost_prob, maxlen=self.prob_size)

    def update(self, blk, hit=False):
        # 1) frequency
        self.window.append(blk)
        self.freq[blk] += 1
        if len(self.window) == self.window.maxlen:
            old = self.window.popleft()
            self.freq[old] -= 1
            if self.freq[old] == 0: del self.freq[old]

        # 2) hits
        if blk in self.prot:
            self.prot.pop(blk); self.prot[blk]=True
            return
        if blk in self.prob:
            self.prob.remove(blk)
            if len(self.prot)>=self.prot_size: self.prot.popitem(last=False)
            self.prot[blk]=True
            return

        # 3) ghost‐hit on probation eviction
        if blk in self.ghost_prob:
            self.ghost_prob.remove(blk)
            self.ghost_hits += 1
            # grow probation fraction
            self.prob_fraction = min(0.5, self.prob_fraction + self.adjust_step)
            self._repartition()
            # admit directly to protected
            if len(self.prot)>=self.prot_size: self.prot.popitem(last=False)
            self.prot[blk]=True
            return

        # 4) cold miss → admission by TinyLFU (freq>=2)
        if self.freq.get(blk,0)>=2:
            # evict from probation if full
            if len(self.prob)>=self.prob_size:
                old = self.prob.popleft()
                self.ghost_prob.append(old)
            self.prob.append(blk)

        # 5) periodic adjustment: shrink probation if too many ghosts
        self.accesses += 1
        if self.accesses>=self.adjust_interval:
            # if >10% of probed ghosts were hit, we keep growing; 
            # otherwise shrink a bit
            rate = self.ghost_hits/self.accesses
            if rate<0.1:
                self.prob_fraction = max(0.05, self.prob_fraction - self.adjust_step)
                self._repartition()
            self.accesses=0; self.ghost_hits=0

    def evict(self, cache):
        # eviction: prefer probation head
        for cand in list(self.prob):
            if cand in cache:
                self.prob.remove(cand)
                return cand
        # else protected LRU
        for cand in self.prot:
            if cand in cache:
                self.prot.pop(cand)
                return cand
        # fallback
        return next(iter(cache))
