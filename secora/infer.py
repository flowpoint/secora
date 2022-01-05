import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

def build_embedding_space(model, data_loader, config, embedding_size=768, feature_prefix=''):
    batch_size = data_loader.batch_size
    dataset_shape = (len(data_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    embedding_space = np.zeros(dataset_shape, dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(config['device'])

            input_ids = batch[feature_prefix + 'input_ids']
            token_type_ids = batch[feature_prefix + 'token_type_ids']
            attention_mask = batch[feature_prefix + 'attention_mask']

            if i == 0:
                model = torch.jit.trace(model.forward, (input_ids, token_type_ids, attention_mask))

            sample_embedding = model(
                    input_ids,
                    token_type_ids,
                    attention_mask)

            embedding_space[i*batch_size:(i+1)*batch_size] = sample_embedding.detach().cpu().numpy()

    return embedding_space


def k_nearest_neighbors(
        model,
        valid_set,
        embedding_size,
        top_k,
        config):

    batch_size = config['infer_batch_size']

    # don't shuffle validation set!
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True)

    dataset_shape = (len(valid_loader)*batch_size, embedding_size)


    code_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='code_', embedding_size=128)
    doc_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='doc_', embedding_size=128)

    similarities = F.cosine_similarity(torch.tensor(code_embedding), torch.tensor(doc_embedding)).detach().cpu().numpy()

    #build the faiss index
    index = faiss.index_factory(embedding_size, 'Flat')
    index.train(code_embedding)
    index.add(code_embedding)

    distances, neighbors = index.search(doc_embedding.astype(np.float32), top_k)

    print(f'simil: {np.mean(similarities)}')
    return distances, neighbors 
