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

    def update(self, training_progress, metric_logger):
        self.epoch_bar.n = training_progress.epoch
        self.shard_bar.n = training_progress.shards

    # try to keep progressbars strictly for progress
    # i.e. no number of batches in progressbars
    # display num batches and num samples in a different way
    def start_training(self):
        self.epoch_bar = tqdm(
            total=self.num_epochs,
            unit=' epoch', 
            desc='training_epoch', 
            smoothing=0.03,
            disable=not self.progress)
        return self.epoch_bar

    def start_epoch(self):
        self.shard_bar = tqdm(
            total=ceil(self.dataset_len/self.num_shards),
            unit=' shard', 
            desc='train_shard', 
            smoothing=0.03,
            disable=not self.progress)

        return self.shard_bar

    def start_shard(self):
        self.step_bar = tqdm(
            total=ceil(self.dataset_len/self.num_steps),
            unit=' steps', 
            desc='train_shard', 
            smoothing=0.03,
            disable=not self.progress)

        return self.shard_bar

    def start_embedding(self, feature_prefix):
        self.embedding_bar = tqdm(
                total=self.valid_len,
                unit=' distractor samples', 
                desc=f'building embeddings: {feature_prefix}', 
                smoothing=0.03,
                disable=not self.progress)
        return self.embedding_bar
