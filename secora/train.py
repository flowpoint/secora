from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses, evaluation
from torch.utils.data import DataLoader

from datasets import load_dataset
import torch

import pandas
import numpy as np


batch_size = 220
device = torch.device('cpu')

# Define your sentence transformer model using CLS pooling
model_name = 'huggingface/CodeBERTa-small-v1'

word_embedding_model = models.Transformer(model_name, max_seq_length=32)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

dataset = load_dataset("code_search_net")

##

train_set = dataset['train']
validation_set = dataset['validation']

train_sentences = train_set['whole_func_string']

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

eval_log_path = "./runs/model_eval"
os.makedirs(eval_log_path, exist_ok=True)

##
corpus = {}
dev_queries = {}
dev_rel_docs = {}

for sample in validation_set:
    qid = sample['func_code_url']
    pid = qid
    dev_queries[qid] = ' '.join(sample['func_documentation_tokens'])
    dev_rel_docs[qid] = set([pid])
    corpus[qid] = ' '.join(sample['func_code_tokens'])

evaluator = evaluation.InformationRetrievalEvaluator(
        dev_queries, 
        corpus, 
        dev_rel_docs, 
        mrr_at_k=range(1,6), 
        corpus_chunk_size=100000,
        ndcg_at_k=range(1,6),
        )

evaluator(model, eval_log_path)

##
# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

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
path = os.path.join(eval_log_path, "Information-Retrieval_evaluation_results.csv")
df = pandas.read_csv(path)
df['cos_sim-MRR@5']
df['dot_score-MRR@5']
df['dot_score-NDCG@5']

