import logging
from time import time 


import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm

from losses import mrr


def build_embedding_space(model, data_loader, config, embedding_size=768, feature_prefix='', device='cpu'):
    rank = dist.get_rank()

    batch_size = data_loader.batch_size
    dataset_shape = (len(data_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    embedding_space = torch.zeros(dataset_shape, dtype=torch.float32, device=rank)
    model.eval()

    if rank == 0:
        bar = tqdm(total=len(data_loader), unit=' batch', desc=f'building embeddings: {feature_prefix}', smoothing=0.03)

    for i, batch in enumerate(data_loader):
        input_ids = batch[feature_prefix + 'input_ids'].to(device)
        token_type_ids = batch[feature_prefix + 'token_type_ids'].to(device)
        attention_mask = batch[feature_prefix + 'attention_mask'].to(device)

        #if i == 0:
        #    model = torch.jit.trace(model.forward, (input_ids, token_type_ids, attention_mask))

        sample_embedding = model(
                input_ids,
                token_type_ids,
                attention_mask)

        # because it's an bi embedding model during distributed training
        sample_embedding = sample_embedding[:,0]
        embedding_space[i*batch_size:(i+1)*batch_size] = sample_embedding.detach()

        if rank == 0:
            bar.update(n=1)

    return embedding_space


def gather(embeddings):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gathered = [torch.zeros_like(embeddings, dtype=torch.float32, device=rank)] * world_size
    dist.all_gather(gathered, embeddings)

    return gathered 


def k_nearest_neighbors(
        query_vectors,
        value_vectors,
        embedding_size,
        top_k):

    rank = dist.get_rank()

    q_gathered = gather(query_vectors)
    v_gathered = gather(value_vectors)

    if rank == 0:
        #build the faiss index
        q_space = torch.cat(q_gathered, 1).to('cpu').numpy()
        v_space = torch.cat(v_gathered, 1).to('cpu').numpy()

        index = faiss.index_factory(embedding_size, 'SQfp16')#Flat')
        index.train(v_space)
        index.add(v_space)

        distances, neighbors = index.search(q_space, top_k)

        return distances, neighbors
    else:
        return None, None


def validate(
        model, 
        valid_loader, 
        config, 
        writer,
        training_progress):
    relevant_ids = range(len(valid_loader))

    with model.no_sync():
        with torch.no_grad():
            code_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='code_', embedding_size=config['embedding_size'], device=rank)
            doc_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='doc_', embedding_size=config['embedding_size'], device=rank)

            dist.barrier()
            distances, neighbors = k_nearest_neighbors(
                    doc_embedding,
                    code_embedding,
                    embedding_size=config['embedding_size'], 
                    top_k=config['top_k'])
                        

    rank = dist.get_rank()
    if rank == 0:
        neighbors_list = [list(n) for n in neighbors]
        score = mrr(list(relevant_ids), neighbors_list)

        i = training_progress.optimizer_step
        writer.add_scalar("mrr/validation", score, i)
        writer.add_scalar("distances/validation", np.mean(distances), i)
        writer.flush()
