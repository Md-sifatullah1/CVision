
import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class ONNXSemanticMatcher:
    def __init__(self, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold

        # Choose an architecture-compatible quantized model.
        machine = os.uname().machine.lower()

        if machine in ("arm64", "aarch64"):
            model_file = "onnx/model_qint8_arm64.onnx"
        else:
            model_file = "onnx/model_quint8_avx2.onnx"

        model_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=model_file,
        )

        tokenizer_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="tokenizer.json",
        )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def similarity(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> float:
        if not requirement.strip() or not cv_evidence.strip():
            raise ValueError("Both texts must be nonempty.")

        encoded = self.tokenizer.encode_batch(
            [requirement, cv_evidence]
        )

        input_ids = np.array(
            [item.ids for item in encoded],
            dtype=np.int64,
        )

        attention_mask = np.array(
            [item.attention_mask for item in encoded],
            dtype=np.int64,
        )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        available_inputs = {
            item.name for item in self.session.get_inputs()
        }

        if "token_type_ids" in available_inputs:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, inputs)
        token_embeddings = outputs[0]

        mask = attention_mask[:, :, None]

        embeddings = (
                (token_embeddings * mask).sum(axis=1)
                / np.maximum(mask.sum(axis=1), 1)
        )

        embeddings /= np.maximum(
            np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )

        return float(embeddings[0] @ embeddings[1])

    def predict(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> int:
        return int(
            self.similarity(requirement, cv_evidence)
            >= self.threshold
        )
