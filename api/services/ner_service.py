import json
import re

import torch

from pipeline.config import ROOT


class NERService:
    def __init__(self):
        self.model = None
        self.model_type = None
        self._load_best_model()

    def _load_best_model(self):
        info_path = ROOT / "best_model_info.json"
        if not info_path.exists():
            print("[ner_service] no best_model_info.json, falling back to crf")
            self.model_type = "crf"
        else:
            with open(info_path) as f:
                info = json.load(f)
            self.model_type = info["model"]

        print(f"[ner_service] loading {self.model_type}")
        loaders = {
            "crf": self._load_crf,
            "bilstm_crf": self._load_bilstm_crf,
            "bert_ner": self._load_bert_ner,
        }
        loaders[self.model_type]()

    def _load_crf(self):
        from pipeline.models.crf_model import CRFModel
        model_path = ROOT / "saved_models" / "crf" / "model.crfsuite"
        self.model = CRFModel()
        self.model.load(model_path)

    def _load_bilstm_crf(self):
        from pipeline.models.bilstm_crf import BiLSTMCRF
        processed = ROOT / "data" / "processed"

        with open(processed / "word2idx.json") as f:
            self.word2idx = json.load(f)
        with open(processed / "tag2idx.json") as f:
            self.tag2idx = json.load(f)
        with open(processed / "idx2tag.json") as f:
            self.idx2tag = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMCRF(
            vocab_size=len(self.word2idx),
            tagset_size=len(self.tag2idx),
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(ROOT / "saved_models" / "bilstm_crf" / "model.pt", weights_only=True, map_location=self.device)
        )
        self.model.eval()

    def _load_bert_ner(self):
        from pipeline.models.bert_ner import BertNER, BertNERTokenizer
        processed = ROOT / "data" / "processed"

        with open(processed / "tag2idx.json") as f:
            self.tag2idx = json.load(f)
        with open(processed / "idx2tag.json") as f:
            self.idx2tag = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = ROOT / "saved_models" / "bert_ner"

        self.bert_tokenizer = BertNERTokenizer.__new__(BertNERTokenizer)
        from transformers import BertTokenizerFast
        self.bert_tokenizer.tokenizer = BertTokenizerFast.from_pretrained(str(model_dir / "tokenizer"))
        self.bert_tokenizer.max_len = 128

        self.model = BertNER("bert-base-uncased", len(self.tag2idx)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", weights_only=True, map_location=self.device)
        )
        self.model.eval()

    def predict(self, text: str) -> tuple[list[str], list[str]]:
        tokens = _tokenize(text)
        if not tokens:
            return [], []

        if self.model_type == "crf":
            tags = self.model.predict_tokens(tokens)
        elif self.model_type == "bilstm_crf":
            tags = self._predict_bilstm(tokens)
        elif self.model_type == "bert_ner":
            tags = self._predict_bert(tokens)

        return tokens, tags

    def _predict_bilstm(self, tokens):
        ids = [self.word2idx.get(t.lower(), 1) for t in tokens]
        length = len(ids)
        pad_len = 128 - length
        ids += [0] * pad_len
        mask = [True] * length + [False] * pad_len

        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        m = torch.tensor([mask], dtype=torch.bool).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(x, m)[0].cpu().tolist()

        return [self.idx2tag.get(str(p), "O") for p in preds[:length]]

    def _predict_bert(self, tokens):
        encoding, word_ids = self.bert_tokenizer.encode_for_inference(tokens)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

        tags = []
        prev_wid = None
        for pid, wid in zip(preds, word_ids):
            if wid is None or wid == prev_wid:
                continue
            tags.append(self.idx2tag.get(str(pid), "O"))
            prev_wid = wid

        return tags[:len(tokens)]

    def extract_entities(self, tokens, tags):
        entities = []
        current_entity = None
        char_offset = 0

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            start = char_offset
            end = start + len(token)

            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "label": tag[2:],
                    "start": start,
                    "end": end,
                }
            elif tag.startswith("I-") and current_entity:
                current_entity["text"] += " " + token
                current_entity["end"] = end
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

            char_offset = end + 1

        if current_entity:
            entities.append(current_entity)

        return entities


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)
