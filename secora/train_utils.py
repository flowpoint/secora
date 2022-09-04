from dataclasses import dataclass, field

import torch
import numpy as np
import random

import grad_cache.functional

def rng_init(seed, deterministic=True):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.enabled = deterministic# benchmark = True

@dataclass
class GCache:
    cache_1: list = field(default_factory=list)
    cache_2: list = field(default_factory=list)
    closures_1: list = field(default_factory=list)
    closures_2: list = field(default_factory=list)


def gradcache_ify(forward1, forward2, loss_fn, optimize_fn, cache):
    ''' the cache is side effected '''
    def cfn1(*args, **kwargs):
        emb, closure = grad_cache.functional.cached(forward1)(*args, **kwargs)
        cache.cache_1.append(emb)
        cache.closures_1.append(closure)
        return emb

    def cfn2(*args, **kwargs):
        emb, closure = grad_cache.functional.cached(forward2)(*args, **kwargs)
        cache.cache_2.append(emb)
        cache.closures_2.append(closure)
        return emb

    def lfn(*args, **kwargs):
        loss = grad_cache.functional.cat_input_tensor(loss_fn)(*args, **kwargs)
        return loss

    def ofn(*args, **kwargs):
        for c, v in zip(cache.closures_1, cache.cache_1):
            c(v)
        for c, v in zip(cache.closures_2, cache.cache_2):
            c(v)

        res = optimize_fn(*args, **kwargs)

        cache.cache_1 = []
        cache.cache_2 = []
        cache.closures_1 = []
        cache.closures_2 = []

        return res

    return cfn1, cfn2, lfn, ofn 
