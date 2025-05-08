# Cache Replacement Policies

## Files

- **policies.py**  
  - LRUPolicy: evicts least recently used block  
  - LFUWithDecayPolicy: evicts least frequently used, with periodic decay  
  - TinyLFUSLRUPolicy: small frequency filter + two‐segment LRU  

- **simulator.py**  
  - CacheSimulator: holds cache set, counts hits/misses, optional next‐line prefetch  
  - test_policy(): runs a policy on a generated workload and reports hit rate & time  
  - Main section loops over workloads (uniform, zipf, cyclic) for cache size 16 you can add more if you want to test other cache sizes  

- **workload.py**  
  - `generate_workload(n_blocks, length, wtype)`: uniform, zipfian, or cyclic streams  
