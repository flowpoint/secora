import logging
from time import time 
import json

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm

from .losses import mrr
from .data import deviceloader


def build_embedding_space(model, data_loader, feature_prefix='', device='cpu', **kwargs):
    embedding_size = model.embedding_size

    batch_size = data_loader.batch_size
    dataset_shape = (len(data_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    embedding_space = torch.zeros(dataset_shape, dtype=torch.float32, device=device)

    show_bar = (device in ['cpu', 0] or not dist.is_initialized()) and kwargs.get('progress', False) == True

    bar = tqdm(
            total=len(data_loader), 
            unit=' batch', 
            desc=f'building embeddings: {feature_prefix}', 
            smoothing=0.03,
            disable=not show_bar)

    for i, batch in enumerate(deviceloader(data_loader, device)):
        model_inputs = (
                batch[feature_prefix + 'input_ids'], 
                batch[feature_prefix + 'attention_mask'])

        #if i == 0:
        #    model = torch.jit.trace(model.forward, model_inputs)

        sample_embedding = model(*model_inputs)

        # because it's an bi embedding model during distributed training
        #sample_embedding = sample_embedding[:,0]
        sample_embedding = sample_embedding
        embedding_space[i*batch_size:(i+1)*batch_size] = sample_embedding.detach()
        bar.update(n=1)

    return embedding_space


def gather(embeddings):
    if dist.is_initialized():
        gathered = [torch.zeros_like(
                embeddings, 
                dtype=torch.float32, 
                device=dist.get_rank())] * dist.get_world_size()
        dist.all_gather(gathered, embeddings)
    else:
        gathered = embeddings

    return gathered 


def k_nearest_neighbors(
        query_vectors,
        value_vectors,
        embedding_size,
        top_k,
        **kwargs):

    if 'device' in kwargs:
        rank = kwargs['device']
        world_size = kwargs['world_size']
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()


    q_gathered = gather(query_vectors)
    v_gathered = gather(value_vectors)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        #build the faiss index
        q_space = torch.cat(q_gathered, -2).to('cpu').numpy().astype(np.float32)
        v_space = torch.cat(v_gathered, -2).to('cpu').numpy().astype(np.float32)

        logger = kwargs['logger']
        logger.debug('building knn index')
        logger.debug(f'q_space: {q_space.shape}')
        logger.debug(f'v_space: {v_space.shape}')
        logger.debug(f'embedding: {embedding_size}')

        #index = faiss.index_factory(embedding_size, 'SQfp16')#Flat')
        index = faiss.IndexFlatIP(embedding_size)
        logger.debug(f'normalize v_space')
        faiss.normalize_L2(v_space)
        logger.debug(f'train v_space')
        index.train(v_space)
        logger.debug(f'add v_space')
        index.add(v_space)

        logger.debug(f'normalize q_space')
        faiss.normalize_L2(q_space)
        logger.debug(f'search knn with q_space, with top_k: {top_k}')
        distances, neighbors = index.search(q_space, top_k)

        return distances, neighbors
    else:
        return None, None


def validate(
        model, 
        valid_loader, 
        config, 
        writer,
        training_progress,
        **kwargs):

    model.eval()
    relevant_ids = range(len(valid_loader))
    rank = dist.get_rank()

    with model.no_sync():
        with torch.no_grad():
            code_embedding = build_embedding_space(model, valid_loader, feature_prefix='code_', device=rank, **kwargs)
            doc_embedding = build_embedding_space(model, valid_loader, feature_prefix='doc_', device=rank, **kwargs)

            dist.barrier()
            distances, neighbors = k_nearest_neighbors(
                    doc_embedding,
                    code_embedding,
                    embedding_size=config['embedding_size'], 
                    top_k=config['top_k'],
                    **kwargs)
                        

    logger = kwargs['logger']
    rank = dist.get_rank()
    dist.barrier()
    if rank == 0:
        neighbors_list = [list(n) for n in neighbors]
        logger.debug('calculate mrr')
        score = float(mrr(list(relevant_ids), neighbors_list))
        logger.debug('score finished')
    else:
        logger.debug('score 0')
        score = float(0.)

    dist.barrier()

    logger.debug('tens')
    tens = torch.tensor(score, requires_grad=False, device=rank, dtype=torch.float32)
    dist.broadcast(tens, src=0)
    torch.cuda.synchronize()
    dist.barrier()
    logger.debug('tens.detach')
    s = tens.detach().to('cpu').numpy()
    dist.barrier()

    if rank == 0:
        logger.debug('create embeddings')
        # show embeddings in tensorboard
        samples = []
        for b in valid_loader:
            for u,l in zip(b['url'], b['language']):
                samples.append(json.dumps({'url':u, 'lang':l}))

        i = training_progress.optimizer_step
        writer.add_embedding(code_embedding, metadata=samples, tag='code', global_step=i)
        writer.add_embedding(doc_embedding, metadata=samples, tag='doc', global_step=i)

        logger.debug('log mrr')
        writer.add_scalar("mrr/validation", score, i)
        logger.debug('log validation')
        writer.add_scalar("distances/validation", np.mean(distances), i)
        writer.flush()

    return s
