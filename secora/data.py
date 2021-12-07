from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("code_search_net")
train_set = dataset['train']
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

valid_set = dataset['validation']
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


