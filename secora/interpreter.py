from transformers import AutoTokenizer, AutoModel
from more_itertools import flatten
from datasets import load_dataset
import numpy as np
import faiss
from pdb import set_trace as bp
from .model import get_model
import argparse

def main(checkpoint_path, emb_path):
    tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    #m = AutoModel.from_pretrained('microsoft/codebert-base')
    embdim = 768
    #m = get_model(checkpoint_path, {"embedding_size": embdim}, device='cpu')
    m = get_model(checkpoint_path, embdim, device='cpu')
    m.eval()

    dataset = load_dataset('code_search_net')['train'].filter(lambda x: x['language'] == 'python')

    #code_emb = np.load('/root/evalout2/code_embedding.npz')
    code_emb = np.load(emb_path)

    index = faiss.IndexFlatIP(embdim)
    faiss.normalize_L2(code_emb)
    index.train(code_emb)
    index.add(code_emb)

    while 1:
        try:
            inp = input("enter a natural language search query: ")
            inpt = tok(inp, return_tensors='pt', return_token_type_ids=True)
            query_emb = m(**inpt).pooler_output[0][:embdim].cpu().detach().numpy()
            print('starting search')
            distances, neighbors = index.search(np.expand_dims(query_emb, 0), 5)
            result = dataset.select(flatten(neighbors))

            print('results:')
            print('\n--------\n--------\n')
            print('\n--------\n'.join([x[:200] for x in result['whole_func_string']]))
            print('\n--------\n--------\n')

        except KeyboardInterrupt:
            pass





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation utility')
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('emb_path', type=str)
    args = parser.parse_args()
    main(args.checkpoint_path, args.emb_path)
