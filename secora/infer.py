import logging
from time import time 
import json
from more_itertools import chunked

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

#from tqdm import tqdm

from secora.losses import mrr
from secora.data import deviceloader, LANGUAGES, get_loader


def build_embedding_space(model, data_loader, embedding_size, feature_prefix='', device='cpu', output_device='cpu', **kwargs):
    ''' the embeddings are calculated on device
    the embedding space is collected on the output_device
    '''

    display = kwargs['display']

    batch_size = data_loader.batch_size
    dataset_shape = (len(data_loader)*batch_size, embedding_size)
    # allocate the dataset_embedding
    embedding_space = torch.zeros(dataset_shape, dtype=torch.float32, device=output_device)
    display.start_embedding(len(data_loader), feature_prefix)

    for i, batch in enumerate(deviceloader(data_loader, device)):
        model_inputs = (
                batch[feature_prefix + 'input_ids'], 
                batch[feature_prefix + 'attention_mask'])

        #if i == 0:
        #    model = torch.jit.trace(model.forward, model_inputs)

        sample_embedding = model(*model_inputs)
        embedding_space[i*batch_size:(i+1)*batch_size] = sample_embedding.detach().to(output_device)
        display.step_embedding()

    return embedding_space


def gather(embeddings):
    if dist.is_initialized():
        gathered = []
        for e in embeddings:
            emb = e.to(dist.get_rank())

            buf = [torch.zeros_like(emb)] * dist.get_world_size()
            dist.all_gather(buf, emb)

            gathered += [x.to('cpu') for x in buf]

        torch.cuda.synchronize()
        dist.barrier()
        return torch.stack(gathered)
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

    #q_gathered = gather(query_vectors)
    #v_gathered = gather(value_vectors)
    q_gathered = query_vectors
    v_gathered = value_vectors

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        #build the faiss index
        #q_space = torch.cat(q_gathered, -2).to('cpu').numpy().astype(np.float32)
        #v_space = torch.cat(v_gathered, -2).to('cpu').numpy().astype(np.float32)
        q_space = q_gathered.numpy().astype(np.float32)
        v_space = v_gathered.numpy().astype(np.float32)

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


def validate_lang(model, lang, valid_set, config, training_progress, num_distractors=1000, **kwargs):
    metriclogger = kwargs['metriclogger']
    embedding_size = config['embedding_size'] 

    lang_set = valid_set.filter(lambda x: x['language'] == lang)
    if len(lang_set) < num_distractors:
        raise RuntimeError(f'not enough samples for validating, got {len(lang_set)} but needs atleasts {num_distractors}')
    rank = dist.get_rank()

    loader = get_loader(lang_set, config['batch_size'], workers=0, dist=dist.is_initialized(), **kwargs)

    c_emb = build_embedding_space(model, loader, embedding_size, feature_prefix='code_', device=rank, **kwargs)
    d_emb = build_embedding_space(model, loader, embedding_size, feature_prefix='doc_', device=rank, **kwargs)

    code_embedding = gather(c_emb)
    doc_embedding = gather(d_emb)

    torch.cuda.synchronize()
    dist.barrier()
    logger = kwargs['logger']

    # run knn on cpu, it uses internal multithreading anyawy
    if rank == 0:
        # crop to multiple of 
        emb_len = code_embedding.shape[0]
        num_chunks = emb_len // num_distractors 

        mrr_distractors = np.zeros([num_chunks])
        dists_distractors = np.zeros([num_chunks])

        c_chunks = (code_embedding[:num_chunks*num_distractors]).reshape([num_chunks, num_distractors, embedding_size])
        d_chunks = (doc_embedding[:num_chunks*num_distractors]).reshape([num_chunks, num_distractors, embedding_size])
        
        relevant_ids = range(num_distractors)
        for i in range(num_chunks):
            logger.debug(f'run knn on chunk: {i} embeddings')
            chunk_dists, neighbors = k_nearest_neighbors(
                    c_chunks[i],
                    d_chunks[i],
                    embedding_size=config['embedding_size'], 
                    top_k=num_distractors,
                    **kwargs)

            neighbors_list = [list(n) for n in neighbors]
            logger.debug('calculate mrr')
            chunk_score = float(mrr(list(relevant_ids), neighbors_list))
            logger.debug('score finished')
            mrr_distractors[i] = chunk_score
            dists_distractors[i] = np.mean(chunk_dists)

        lang_score = np.mean(mrr_distractors)
        lang_dists = np.mean(dists_distractors)

        logger.debug('create embeddings')
        # show embeddings in tensorboard
        samples = []
        for b in loader:
            for u,l in zip(b['url'], b['language']):
                samples.append(json.dumps({'url':u, 'lang':l}))

        global_step = training_progress.optimizer_step
        metriclogger.add_embedding(code_embedding, metadata=samples, tag=f'{lang}_code', global_step=global_step)
        metriclogger.add_embedding(doc_embedding, metadata=samples, tag=f'{lang}_doc', global_step=global_step)

        logger.debug('log mrr')
        metriclogger.add_scalar(f"mrr/validation/{lang}", lang_score, global_step)
        logger.debug('log validation')
        metriclogger.add_scalar(f"distances/validation/{lang}", lang_dists, global_step)
        metriclogger.flush()

        return lang_score, lang_dists
    else:
        return None, None


def validate(
        model, 
        valid_set, 
        config, 
        training_progress,
        **kwargs):

    model.eval()
    rank = dist.get_rank()

    logger = kwargs['logger']
    metriclogger = kwargs['metriclogger']

    if config['languages'] == ['all']:
        langs = LANGUAGES
    else: 
        langs = config['languages']

    scores = []
    dists = []

    if kwargs.get('debug', False) == True:
        num_distractors = 4
    else:
        num_distractors = 1000

    with model.no_sync():
        with torch.no_grad():
            for lang in langs:
                l_score, l_dists = validate_lang(model, lang, valid_set, config, training_progress, num_distractors=num_distractors, **kwargs)
                scores.append(l_score)
                dists.append(l_dists)

    dist.barrier()
    if rank == 0:
        m_avg_score = np.mean(scores)
        m_avg_dist = np.mean(scores)

        global_step = training_progress.optimizer_step

        logger.debug('log mrr')
        metriclogger.add_scalar(f"mrr/validation/lang_avg", m_avg_score, global_step)
        logger.debug('log validation')
        metriclogger.add_scalar(f"distances/validation/lang_avg", m_avg_dist, global_step)
        metriclogger.flush()

    dist.barrier()

    return m_avg_score
