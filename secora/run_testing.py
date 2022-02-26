from collections import OrderedDict
from more_itertools import chunked
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from pdb import set_trace as bp

import argparse
import torch
from datasets import load_dataset
from . import model
from . import infer
from . import data
import numpy as np
from . import losses

import faiss


def get_model(checkpoint_path, basemodel, embsize, device, **kwargs):
    st = torch.load(checkpoint_path, map_location=device)[0]
    m = model.BiEmbeddingModel(basemodel, embsize, **kwargs).to(device)
    st2 = OrderedDict()
    for k in st.keys():
        v = st[k]
        #st2[k.removeprefix('module.')] = v
        st2[k.replace('module.m.', '', 1)] = v
        
    m.load_state_dict(st2)
    return m



def test(
        model, 
        valid_loader, 
        device,
        **kwargs):

    model.eval()
    relevant_ids = range(len(valid_loader))

    with torch.no_grad():
        code_embedding = infer.build_embedding_space(model, valid_loader, None, feature_prefix='code_', embedding_size=768, device=device, progress=True, **kwargs)
        doc_embedding = infer.build_embedding_space(model, valid_loader, None, feature_prefix='doc_', embedding_size=768, device=device, progress=True, **kwargs)

        q_space = doc_embedding.cpu().numpy().astype(np.float32)
        v_space = code_embedding.cpu().numpy().astype(np.float32)

        index = faiss.IndexFlatIP(768)
        faiss.normalize_L2(v_space)
        index.train(v_space)
        index.add(v_space)

        faiss.normalize_L2(q_space)
        distances, neighbors = index.search(q_space, 1000)

        neighbors_list = [list(n) for n in neighbors]
        score = float(losses.mrr(list(relevant_ids), neighbors_list))
        return score


def get_loader(dataset, batch_size, **kwargs):
    loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=6, 
            multiprocessing_context='spawn',
            persistent_workers=True, 
            sampler=None)

    return loader

def main(output_dir, config_path, batches_path, checkpoint_path, device):
    m = get_model(checkpoint_path, model.BaseModel.CODEBERT, 768, device)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model_name'] = model.BaseModel.CODEBERT
    config['languages'] = ['all']

    testset = data.preprocess_split(data.DataSplit.TEST, config, progress=True)
    testset.shuffle()
    
    scores = []
    with open(output_dir, "w") as f:
        f.write('begin\n')
        for chunk in tqdm(chunked(testset, 1000)):
            s = test(m, get_loader(chunk, 10), device)
            scores.append(s)
            f.write(f'{s}\n')

        avg = sum(scores)/len(scores)
        #print(f"mrr: {avg}")
        f.write(f'final: {avg}\n')

def export_dataset(output_dir, config_path, batches_path, checkpoint_path, device):
    m = get_model(checkpoint_path, model.BaseModel.CODEBERT, 768, device)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    testset = data.preprocess_split(data.DataSplit.TEST, config, progress=True)
    testset.shuffle()
    testset.save_to_disk(output_dir)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation utility')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('batches_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    args.run_name = 'evaluate'

    torch.set_num_threads(10)
    #main(args.output_dir, args.config_path, args.batches_path, args.checkpoint_path, args.device)
    export_dataset(args.output_dir, args.config_path, args.batches_path, args.checkpoint_path, args.device)
