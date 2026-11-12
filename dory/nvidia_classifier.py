"""Custom multi-head model copied from the nvidia/prompt-task-and-complexity-classifier card."""
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer

REPO = "nvidia/prompt-task-and-complexity-classifier"


class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        m = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * m, 1) / torch.clamp(m.sum(1), min=1e-9)


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            top2_probs = torch.softmax(preds, dim=1).gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()
            top2_strings = [[self.task_type_map[str(idx)] for idx in s] for s in top2]
            top2_prob_rounded = [[round(v, 3) for v in s] for s in top2_prob]
            for i, s in enumerate(top2_prob_rounded):
                if s[1] < 0.1:
                    top2_strings[i][1] = "NA"
            return ([s[0] for s in top2_strings], [s[1] for s in top2_strings],
                    [s[0] for s in top2_prob_rounded])
        preds = torch.softmax(preds, dim=1)
        weights = np.array(self.weights_map[target])
        scores = np.sum(np.array(preds.detach().cpu()) * weights, axis=1) / self.divisor_map[target]
        scores = [round(v, decimal) for v in scores]
        if target == "number_of_few_shots":
            scores = [x if x >= 0.05 else 0 for x in scores]
        return scores

    def process_logits(self, logits):
        result = {}
        t1, t2, tp = self.compute_results(logits[0], target="task_type")
        result["task_type_1"], result["task_type_2"], result["task_type_prob"] = t1, t2, tp
        for i, target in [(1, "creativity_scope"), (2, "reasoning"), (3, "contextual_knowledge"),
                          (4, "number_of_few_shots"), (5, "domain_knowledge"),
                          (6, "no_label_reason"), (7, "constraint_ct")]:
            result[target] = self.compute_results(logits[i], target=target)
        result["prompt_complexity_score"] = [
            round(0.35 * cr + 0.25 * re + 0.15 * co + 0.15 * dk + 0.05 * ck + 0.05 * fs, 5)
            for cr, re, co, dk, ck, fs in zip(
                result["creativity_scope"], result["reasoning"], result["constraint_ct"],
                result["domain_knowledge"], result["contextual_knowledge"],
                result["number_of_few_shots"])]
        return result

    def forward(self, batch):
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pooled = self.pool(out.last_hidden_state, batch["attention_mask"])
        return self.process_logits([self.heads[k](pooled) for k in range(len(self.target_sizes))])


def load_classifier(device=None):
    """Returns (model, tokenizer, device). Weights download on first call."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(REPO)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = CustomModel(target_sizes=cfg.target_sizes, task_type_map=cfg.task_type_map,
                        weights_map=cfg.weights_map, divisor_map=cfg.divisor_map
                        ).from_pretrained(REPO)
    return model.to(device).eval(), tok, device
