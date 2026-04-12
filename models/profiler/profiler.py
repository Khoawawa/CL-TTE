import time

import torch

class BlockTimer:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.times = {}
    
    def start(self, key):
        if self.use_cuda:
            torch.cuda.synchronize()
        
        self._t0 = time.perf_counter()
        self._key = key
    def stop(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        
        dt = time.perf_counter() - self._t0
        self.times[self._key] += self.times.get(self._key, 0) + dt
        
    def report(self):
        total = sum(self.times.values())
        print("\n=== Profiling ===")
        for k, v in self.times.items():
            print(f"{k:25s}: {v*1000:.3f} ms ({v/total*100:.1f}%)")
        print(f"{'TOTAL':25s}: {total*1000:.3f} ms\n")