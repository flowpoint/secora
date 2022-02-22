import pytest

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import tempfile


@pytest.mark.slow
def test_onnx_export():
    m = AutoModel.from_pretrained('microsoft/codebert-base').base_model
    tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def toke(s):
        dummy = tok(s, padding='max_length', truncation=True, return_tensors='pt', return_token_type_ids=True)

        d_input_ids = dummy['input_ids']
        d_attention_mask = dummy['attention_mask']

        return d_input_ids, d_attention_mask

    s = 'if __name__ == <mask>:'

    inp = toke(s)
    out = m(*inp)

    with tempfile.NamedTemporaryFile() as f:
        torch.onnx.export(
                m, 
                inp, 
                f.name,
                opset_version=11,
                export_params=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state', 'pooler_output'])


        import onnx
        onnx_model = onnx.load(f.name)
        onnx.checker.check_model(onnx_model)

        import onnxruntime
        ort_session = onnxruntime.InferenceSession(f.name)

        dummy = tok(s, padding='max_length', truncation=True, return_token_type_ids=True)

        d_input_ids = dummy['input_ids']
        d_attention_mask = dummy['attention_mask']

        ort_inp = {arg.name: np.array([dummy[arg.name]]) for arg in ort_session.get_inputs()}

        output_names = [o.name for o in ort_session.get_outputs()]
        ort_os = ort_session.run(None, ort_inp)
        ort_outs = {n: v for n,v in zip(output_names, ort_os)}

        tok.decode(np.argmax(ort_outs['last_hidden_state'], axis=-1)[0])
