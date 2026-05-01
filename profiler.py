from models.main_model import Cl_TTE as CLTTE

model = CLTTE(
    d_model=128,
    nhead=8,
    seq_layer=2
)
def count_params_detailed(model):
    stats = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        module = name.split('.')[0]  # top-level
        submodule = ".".join(name.split('.')[:2])  # one level deeper

        stats.setdefault(module, 0)
        stats[module] += param.numel()

        stats.setdefault(submodule, 0)
        stats[submodule] += param.numel()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== Top-level ===")
    for k, v in sorted(stats.items()):
        if "." not in k:
            print(f"{k:25s}: {v/1e6:.3f} M ({v/total*100:.1f}%)")

    print("\n=== Sub-modules ===")
    for k, v in sorted(stats.items()):
        if "." in k:
            print(f"{k:25s}: {v/1e6:.3f} M ({v/total*100:.1f}%)")

    print(f"\nTOTAL: {total/1e6:.3f} M")
    
count_params_detailed(model)

import torch
import time
from collections import defaultdict

class ModuleProfiler:
    def __init__(self, model):
        self.model = model
        self.fwd_times = defaultdict(float)
        self.bwd_times = defaultdict(float)
        self.handles = []

    def _fwd_pre_hook(self, name):
        def hook(module, input):
            torch.cuda.synchronize()
            module.__start_time = time.time()
        return hook

    def _fwd_hook(self, name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            elapsed = time.time() - module.__start_time
            self.fwd_times[name] += elapsed
        return hook

    def _bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            torch.cuda.synchronize()
            start = time.time()

            def _end_hook(*_):
                torch.cuda.synchronize()
                self.bwd_times[name] += time.time() - start

            # attach to output grad
            if isinstance(grad_output, tuple):
                for g in grad_output:
                    if g is not None:
                        g.register_hook(_end_hook)
                        break
        return hook

    def attach(self):
        for name, module in self.model.named_children():
            self.handles.append(module.register_forward_pre_hook(self._fwd_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._fwd_hook(name)))
            self.handles.append(module.register_full_backward_hook(self._bwd_hook(name)))

    def clear(self):
        for h in self.handles:
            h.remove()

    def report(self, iters=1):
        print("\n=== Forward Time per Module ===")
        for k, v in self.fwd_times.items():
            print(f"{k:20s}: {v/iters:.6f}s")

        print("\n=== Backward Time per Module ===")
        for k, v in self.bwd_times.items():
            print(f"{k:20s}: {v/iters:.6f}s")
            
prof = ModuleProfiler(model)
prof.attach()
from utils.prepare import load_datadict, load_datadoct_pre
import os
import json
with open(f'{os.path.dirname(__file__)}/utils/data_config.json', 'r') as f:
    data_config = json.load(f)['hcm']
args = 
load_datadoct_pre(args)
# warmup (avoid cache / kernel init noise)
for _ in range(10):
    t, l = model(inputs, y_true)
    loss = t.mean() + (l if l is not None else 0)
    loss.backward()
    model.zero_grad()

# actual measurement
iters = 20
for _ in range(iters):
    t, l = model(inputs, y_true)
    loss = t.mean() + (l if l is not None else 0)
    loss.backward()
    model.zero_grad()

prof.report(iters)
prof.clear()