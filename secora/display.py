from tqdm import tqdm
import datasets

class Display:
    ''' handles printing to stderr, and progressbars 
    allows the use of different formatting backends, 
    by default: tqdm and print
    also allows for printing mutiple progressmeters for different training blocks
    '''
    def __init__(self, show_progress, backend='tqdm', rank=0):
        self.progress = show_progress
        self.rank = rank

        if show_progress:
            datasets.enable_progress_bar()
        else:
            datasets.disable_progress_bar()

    def start_shard(self, shard_len):
        self.shard_bar = tqdm(
            total=shard_len,
            unit=' batch', 
            desc='train_shard', 
            smoothing=0.03,
            disable=not self.progress)

    def step_shard(self, n=1):
        self.shard_bar.update(n)

    def start_embedding(self, valid_len, feature_prefix):
        self.embedding_bar = tqdm(
                total=valid_len,
                unit=' batch', 
                desc=f'building embeddings: {feature_prefix}', 
                smoothing=0.03,
                disable=not self.progress)

    def step_embedding(self, n=1):
        self.embedding_bar.update(n)
