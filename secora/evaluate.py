import csv
import sys
import argparse
import torch
import os
from collections import OrderedDict

from more_itertools import flatten, chunked

from .infer import *
from .model import *
from .data import *
from . import data
from .config import load_config, overwrite_config

from datasets import Dataset


csv.field_size_limit(sys.maxsize)

def export_predictions(preds, queries):
    with open('predictions.csv', 'w') as csvfile:

        fieldnames = ['query', 'language', 'url']
        cwriter = csv.DictWriter(csvfile,fieldnames)
        cwriter.writeheader()

        for q, preds in zip(queries, preds):
            for p in preds:
                language = p['language']
                url = p['url']

                cwriter.writerow(
                    {'language': language,
                        'query': q['query'],
                        'url': url
                    })

def get_model(checkpoint_path, config, device):
    st = torch.load(checkpoint_path, map_location=device)[0]
    model = BiEmbeddingModelCuda(BaseModel.CODEBERT, config['embedding_size'], AMP.FP16).to(device)
    #model = EmbeddingModel(BaseModel.CODEBERT, config['embedding_size']).to(device)
    st2 = OrderedDict()

    for k in st.keys():
        v = st[k]
        #st2[k.removeprefix('module.')] = v
        st2[k.replace('module.m.', '', 1)] = v
        
    #st2['m.pooling.weight'] = torch.zeros([768,768])
    #st2['m.pooling.bias'] = torch.zeros([768])
    #st2.pop('m.pooling.bias')

    model.m.load_state_dict(st2)
    return model

def export_code_embedding(test_set, checkpoint_path, path, device, config):
    rank = 0

    loader = DataLoader(
            test_set, 
            batch_size=config['batch_size'], 
            shuffle=False,
            drop_last=True, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            num_workers=6, 
            multiprocessing_context='spawn',
            persistent_workers=False)


    with torch.no_grad():
        code_embedding = build_embedding_space(model, loader, config, feature_prefix='code_', embedding_size=config['embedding_size'], device=device, progress=True).to('cpu').detach().numpy()
   
    with open(path, 'wb') as f:
        np.save(f, code_embedding)



def get_q_emb(model, query_set, config, device):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'].value)

    def tokenize_fn(batch):
        return tokenizer(
                batch['query'], 
                padding='max_length', 
                max_length=config['max_input_tokens'], 
                truncation=True, 
                return_token_type_ids=True)

    query_set = query_set.map(tokenize_fn, batched=True, num_proc=config['preprocess_cores'])
    query_set.set_format(type='torch', columns=set(query_set.column_names) - {'query', 'language'}, output_all_columns=True)

    loader = DataLoader(
            query_set, 
            batch_size=1,
            shuffle=False,
            drop_last=False, 
            pin_memory=True, 
            # workers need to use the spawn or forkserver method in a distributed setting
            persistent_workers=False)

    q_emb = build_embedding_space(model, loader, config, embedding_size=config['embedding_size'], device=device, progress=True).to('cpu').detach().numpy()

    return q_emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation utility')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('queries_csv', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    args.run_name = 'evaluate'

    top_k = args.top_k
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    checkpoint_path = args.checkpoint_path

    if args.device == 'auto' and torch.cuda.is_available():
        dname = 'cuda'
    else:
        dname = args.device
    device = torch.device(dname)


    config = load_config(args.config_path)
    config = overwrite_config(args, config)

    #config['languages'] = ['python', 'java', 'javascript', 'php', ]
    #config['languages'] = ['all']
    config['model_name'] = BaseModel.CODEBERT

    model = get_model(checkpoint_path, config, device)

    test_set = preprocess_split(data.DataSplit.EVAL, config, progress=True)

    p1 = os.path.join(outdir, 'code_embedding.npz')
    if not os.path.isfile(p1):
        export_code_embedding(test_set, checkpoint_path, p1, device, config)

    with open(p1, 'rb') as f:
        #code_embedding = np.load(p1, allow_pickle=True)
        code_embedding = np.load(p1,).astype(np.float32)

    query_set = Dataset.from_csv(args.queries_csv, keep_in_memory=True)
    q_emb = get_q_emb(model, query_set, config, device).astype(np.float32)

    #index = faiss.index_factory(config['embedding_size'], 'SQfp16')#Flat')
    index = faiss.IndexFlatIP(config['embedding_size'])
    faiss.normalize_L2(code_embedding)
    index.train(code_embedding)
    index.add(code_embedding)

    faiss.normalize_L2(q_emb)
    distances, neighbors = index.search(q_emb, top_k)
    neighbors_list = [list(n) for n in neighbors]
    neigh_set = test_set.select(flatten(neighbors_list))
    results = chunked(neigh_set, top_k, strict=True)
    export_predictions(results, list(query_set))

