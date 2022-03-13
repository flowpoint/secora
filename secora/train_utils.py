from grad_cache.functional import cached, cat_input_tensor
from tqdm import tqdm
from time import time
import logging

class GCache:
    def __init__(self, temp, loss_fn_):
        self.temp = temp
        self.loss_fn_ = loss_fn_

        self.cache_1: list = []
        self.cache_2: list = []
        self.closures_1: list = []
        self.closures_2: list = []

    def reset(self):
        self.cache_1 = []
        self.cache_2 = []
        self.closures_1 = []
        self.closures_2 = []

    @cached
    def call_model(self, model, model_inputs):
        return model(*model_inputs)

    @cat_input_tensor
    def loss_fn(self, x, y):
        return self.loss_fn_(x, y, temp=self.temp)


class ProgressDisplay:
    ''' logs the livelyness and progress of a training task '''

    def __init__(self, training_progress, len_, enabled, **kwargs):
        self.enabled = enabled
        self.heartbeat = time()
        self.training_progress = training_progress

        show_bar = kwargs.get('progress', False) and self.enabled == 0
        print('--------')
        print(len_)
        print('--------')

        self.bar = tqdm(
            total=len_,
            unit=' batch', 
            desc='train_shard', 
            smoothing=0.03,
            disable=not show_bar)

    def update(self):
        if self.enabled and time() - self.heartbeat > 10:
            self.bar.clear()
            e = self.training_progress.epoch
            s = self.training_progress.shard
            #b = training_progress.shard
            batch = self.training_progress.batch
            logger = logging.getLogger('secora')
            logger.info(f"heartbeat: training: epoch: {e} shard: {s} batch: {batch}")
            self.heartbeat = time()

        self.bar.update(n=1)

    def reset(self):
        self.bar.reset()
