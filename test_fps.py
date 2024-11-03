import time

import torch

from model.MVANet import MVANet

data = torch.randn(1, 3, 1024, 1024, device='cuda', dtype=torch.float32)

model = MVANet().eval().cuda()  
for _ in range(10):
    model(data)

torch.cuda.synchronize()
start_time = time.perf_counter()
for _ in range(100):
    model(data)
torch.cuda.synchronize()
print(f"FPS: {100 / (time.perf_counter() - start_time):.03f}")
