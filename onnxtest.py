#import onnxruntime as ort
import onnx

m = onnx.load('model.onnx')

onnx.checker.check_model(m)


##
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession('model.onnx')

input_ids = np.random.randint(0, 700, size=(2,266))
token_type_ids = np.zeros((2,266), dtype=np.int64)
attention_mask = np.ones((2,266), dtype=np.int64)
##
outputs = ort_session.run(None,
        {'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
        })

##
from transformers import AutoTokenizer
model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
