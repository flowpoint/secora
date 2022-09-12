from math import ceil
from tqdm import tqdm
import datasets


class Display:
    ''' handles printing to stderr, progressbars 
    displays training progress and some metrics
    allows the use of different formatting backends, 
    by default: tqdm and print
    also allows for printing mutiple progressmeters for different training blocks
    '''
    def __init__(self, show_progress, 
            backend='tqdm', rank=0):
        self.progress = show_progress
        self.rank = rank

        self.epoch_bar = None
        self.shard_bar = None
        self.step_bar = None
        self.embedding_bar = None

        if show_progress:
            datasets.enable_progress_bar()
        else:
            datasets.disable_progress_bar()

    def set_total(self,
            dataset_len, 
            valid_len,
            num_epochs,
            num_shards,
            num_steps,
            ):
        self.dataset_len = dataset_len
        self.valid_len = valid_len
        self.num_epochs = num_epochs
        self.num_shards = num_shards
        self.num_steps = num_steps

        self.epoch_bar = tqdm(
            total=self.num_epochs,
            position=0,
            unit=' epoch', 
            desc='training_epoch', 
            smoothing=0.03,
            disable=not self.progress)

        self.shard_bar = tqdm(
            total=self.num_shards,
            position=1,
            unit=' shard', 
            desc='--train_shard', 
            smoothing=0.03,
            disable=not self.progress)

        self.step_bar = tqdm(
            total=self.num_steps,
            position=2,
            unit=' steps', 
            desc='----train_step', 
            smoothing=0.03,
            disable=not self.progress)

        self.embedding_bar = tqdm(
            total=self.valid_len,
            position=3,
            unit=' distractor samples', 
            desc=f'----building embeddings', 
            smoothing=0.03,
            disable=not self.progress)


    def update(self, training_progress, embedding_step=0):
        self.epoch_bar.n = training_progress.epoch +1 
        self.shard_bar.n = training_progress.shard +1 
        self.step_bar.n = training_progress.step +1 

        self.embedding_bar.n = embedding_step

        self.epoch_bar.refresh()
        self.shard_bar.refresh()
        self.step_bar.refresh()

        self.embedding_bar.refresh()


    # try to keep progressbars strictly for progress
    # i.e. no number of batches in progressbars
    # display total num batches and total num samples in a different way
    def start_training(self):
        self.epoch_bar.reset()

    def start_epoch(self):
        self.shard_bar.reset()

    def start_shard(self):
        self.step_bar.reset()
        self.embedding_bar.reset()

    def start_embedding(self):
        self.embedding_bar.reset()

    def close(self):
        if self.epoch_bar:
            self.epoch_bar.close()
        if self.shard_bar:
            self.shard_bar.close()
        if self.step_bar:
            self.step_bar.close()
        if self.embedding_bar:
            self.embedding_bar.close()
