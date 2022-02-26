from grad_cache.functional import cached, cat_input_tensor

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

bert = torch.nn.Linear(10,1)

@cached
def  call_model(model, input):
    return model(input)

@cat_input_tensor
def  contrastive_loss(x, y):
    return torch.sum(x-y)


cache_x = []
cache_y = []
closures_x = []
closures_y = []

for step, sub_batch in enumerate(torch.ones([10,10])):
    xx = yy = sub_batch
    rx, cx = call_model(bert, xx)
    ry, cy = call_model(bert, yy)
    
    cache_x.append(rx)
    cache_y.append(ry)
    closures_x.append(cx)
    closures_y.append(cy)
    
    if (step + 1) % 6 == 0:
        loss = contrastive_loss(cache_x, cache_y)
        loss.backward()
        
    for f, r in zip(closures_x, cache_x):
        f(r)
    for f, r in zip(closures_y, cache_y):
        f(r)

    cache_x = []
    cache_y = []
    closures_x = []
    closures_y = []
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
