import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

def build_embedding_space(model, data_loader, config, embedding_size=768, feature_prefix='', device='cpu'):
    batch_size = data_loader.batch_size
    dataset_shape = (len(data_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    embedding_space = np.zeros(dataset_shape, dtype=np.float32)

    model.eval()

    bar = tqdm(total=len(data_loader), unit=' batch', desc=f'building embeddings: {feature_prefix}', smoothing=0.03)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch[feature_prefix + 'input_ids'].to(device, non_blocking=True)
            token_type_ids = batch[feature_prefix + 'token_type_ids'].to(device, non_blocking=True)
            attention_mask = batch[feature_prefix + 'attention_mask'].to(device, non_blocking=True)

            #if i == 0:
            #    model = torch.jit.trace(model.forward, (input_ids, token_type_ids, attention_mask))

            sample_embedding = model(
                    input_ids,
                    token_type_ids,
                    attention_mask)

            embedding_space[i*batch_size:(i+1)*batch_size] = sample_embedding.detach().cpu().numpy()
            bar.update(n=1)

    return embedding_space


def k_nearest_neighbors(
        model,
        valid_loader,
        embedding_size,
        top_k,
        config,
        device='cpu'):

    batch_size = config['infer_batch_size']

    dataset_shape = (len(valid_loader)*batch_size, embedding_size)

    code_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='code_', embedding_size=config['embedding_size'])
    doc_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='doc_', embedding_size=config['embedding_size'])

    similarities = F.cosine_similarity(torch.tensor(code_embedding), torch.tensor(doc_embedding)).detach().cpu().numpy()

    #build the faiss index
    index = faiss.index_factory(embedding_size, 'Flat')
    index.train(code_embedding)
    index.add(code_embedding)

    distances, neighbors = index.search(doc_embedding.astype(np.float32), top_k)

    return distances, neighbors, similarities
