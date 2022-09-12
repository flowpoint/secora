import os
from torch.utils.tensorboard import SummaryWriter

'''
present a uniform api for logging metrics
allows to switch between tensorboard, wnb, ...
andor extend the metric logging fast

this also helps minimize the metric logging api surface
'''

class MetricLogger:
    def __init__(self, config, rank):
        if rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(config['logdir'], config['training_run_id']), flush_secs=30)

    def add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args,**kwargs)

    def add_embedding(self, *args, **kwargs):
        self.writer.add_embedding(*args,**kwargs)

    def add_hparams(self, *args, **kwargs):
        self.writer.add_hparams(*args, **kwargs)
        self.writer.flush()

    def flush(self, *args, **kwargs):
        self.writer.flush(*args, **kwargs)
