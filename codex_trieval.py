from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses, evaluation
from torch.utils.data import DataLoader

from datasets import load_dataset
import torch

import pandas
import numpy as np

#import pdb
#from pdb import set_trace as bp

##

#device = torch.device('cpu')

batch_size = 220
device = torch.device('cuda')

# Define your sentence transformer model using CLS pooling
#model_name = 'distilroberta-base'
model_name = 'huggingface/CodeBERTa-small-v1'

word_embedding_model = models.Transformer(model_name, max_seq_length=32)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

dataset = load_dataset("code_x_glue_tc_nl_code_search_adv")

##

train_set = dataset['train']#.select(range(100))
validation_set = dataset['validation']#.select(range(10000))

train_sentences = map(lambda x: x['original_string'], train_set)

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

##

corpus = {}
dev_queries = {}
dev_rel_docs = {}

for sample in validation_set:
    qid = sample['id']
    pid = qid
    #dev_queries[qid] = ' '.join(sample['code_tokens'])
    #dev_rel_docs[qid] = set(' '.join(sample['code_tokens']))

    dev_queries[qid] = ' '.join(sample['docstring_tokens'])
    dev_rel_docs[qid] = set([pid])

    corpus[qid] = ' '.join(sample['code_tokens'])


# todo: verify that evaluation is correct
evaluator = evaluation.InformationRetrievalEvaluator(
        dev_queries, 
        corpus, 
        dev_rel_docs, 
        #mrr_at_k=[10,20,40,50,80,100], 
        mrr_at_k=range(6), 
        corpus_chunk_size=100000,
        #ndcg_at_k=range(5),
        #accuracy_at_k=range(5),
        #precision_recall_at_k=range(5),
        #map_at_k=range(5),
        )
##

#bp()
#model.encode(corpus)
evaluator(model, "/tmp/model_eval",)

##
# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method
# todo: verify that evaluator actually runs
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=100,
)

model.save('output/simcse-model')

##

# loads_evaluator results
path = '/tmp/model_eval/Information-Retrieval_evaluation_results.csv'
df = pandas.read_csv(path)
df['cos_sim-MRR@5']
#df['dot_score-MRR@5']

##
