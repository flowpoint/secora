from grad_cache.functional import cached, cat_input_tensor
from tqdm import tqdm
from time import time

class GCache:
    def __init__(self, temp):
        self.temp = temp

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
        return contrastive_loss(x, y, temp=self.temp)


class ProgressDisplay:
    ''' logs the livelyness and progress of a training task '''

    def __init__(self, training_progress, len_, **kwargs):
        self.rank = dist.get_rank()
        self.heartbeat = time()
        self.training_progress = training_progress

        show_bar = kwargs.get('progress', False) and self.rank == 0
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
        if self.rank == 0 and time() - self.heartbeat > 10:
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
