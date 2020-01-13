from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.span_extractors import EndpointSpanExtractor, SpanExtractor
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric
from torch import nn, Tensor

# from components.data.metrics.instance_wise_accuracy import InstanceWiseCategoricalAccuracy
from torch.nn import CrossEntropyLoss

from al2_implementation.ner_as_qa_f1_measure import NerAsQaSpanF1

TensorDict = Dict[str, torch.Tensor]

# TODO: support nested entities
@Model.register("ner_as_mrc")
class NerAsQaModel(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        vocab: Vocabulary,
        metrics: Dict[str, Metric],
        # hidden_dim: int = 128,
        # dropout: float = 0.1,
        # f1_average: str = "micro",
        # # Loss-specific params
        # target_namespace: str = "labels",
        # target_label: Optional[str] = None,
        # smooth: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        hidden_dim = self.encoder.get_output_dim()
        self.span_starts = nn.Linear(hidden_dim, 2)
        self.span_ends = nn.Linear(hidden_dim, 2)
        self._loss_fn = CrossEntropyLoss()
        metrics = metrics or {}
        metrics.update({
            "start_accuracy": CategoricalAccuracy(),
            "end_accuracy": CategoricalAccuracy(),
            "f1m": F1Measure(positive_label=1),
            "span_f1": NerAsQaSpanF1()
        })
        self.metrics = metrics
        initializer(self)

    def forward(
        self,
        context: TensorDict,  # [B , L , N]
        answer_starts: Tensor = None,  # [B , L]
        answer_ends: Tensor = None,  # [B , L]
        **kwargs,
    ) -> Dict:
        # B -- batch_size
        # L -- number of lines in batch
        # N -- number of tokens in line
        # E -- token embedding dim

        if answer_starts is None and answer_ends is not None:
            raise RuntimeError(f'Answer start ans answer ends must be provided simultaneously')
        if answer_ends is None and answer_starts is not None:
            raise RuntimeError(f'Answer start ans answer ends must be provided simultaneously')
        mask = get_text_field_mask(context)  # [B, L]
        embeddings = self.word_embeddings(context)
        batch_size = embeddings.size(1)
        encoded_lines = self.encoder(embeddings, mask=mask)
        start_logits = self.span_starts(encoded_lines)
        end_logits = self.span_ends(encoded_lines)

        output = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "mask": mask,
        }

        if answer_starts is not None:

            # for metric in self.metrics.values():
            self.metrics['start_accuracy'](start_logits, answer_starts)
            self.metrics['end_accuracy'](end_logits, answer_starts)
            self.metrics["f1m"](start_logits, answer_starts)
            self.metrics["f1m"](end_logits, answer_ends)
            self.metrics['span_f1'](start_logits, end_logits, answer_starts, answer_ends, [meta['type'] for meta in kwargs['meta']])
            start_loss = self._loss_fn(start_logits.view(-1, 2), answer_starts.flatten())
            end_loss = self._loss_fn(end_logits.view(-1, 2), answer_ends.flatten())
            loss = start_loss + end_loss
            output['loss'] = loss
        return output

    # def decode(self, output_dict: Dict[str, torch.Tensor]):
    #     """Get argmax over logits and convert to tags"""
    #     logits = output_dict["logits"]
    #     argmax = logits.argmax(dim=-1)
    #     tags = [
    #         self.vocab.get_token_from_index(idx.item(), self._target_namespace)
    #         for idx in argmax
    #     ]
    #     output_dict["tags"] = tags
    #     return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        for metric_name, metric in self.metrics.items():
            metrics = metric.get_metric(reset)
            if isinstance(metrics, float):
                metrics_to_return[metric_name] = metrics
            elif isinstance(metrics, tuple) and len(metrics) == 3:
                # handle case of fmeasure (for example)
                metrics_to_return["precision"], metrics_to_return[
                    "recall"
                ], metrics_to_return["f1-measure"] = metrics
            elif isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, float):
                        # simple case that handles averaged f measure
                        metrics_to_return[k] = v
                    elif isinstance(v, list):
                        # when f measure is not averaged
                        for i, m in enumerate(v):
                            label = self.vocab.get_token_from_index(
                                i, self._target_namespace
                            )
                            metrics_to_return[k + "_" + label] = m

                # metrics_to_return.update(metrics)
            else:
                raise RuntimeError(f"Metric {metric_name} cannot be displayed")
        return metrics_to_return
