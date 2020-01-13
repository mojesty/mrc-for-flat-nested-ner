from typing import Dict, List, Optional, Set, Callable, Tuple
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
        bio_tags_to_spans,
        bioul_tags_to_spans,
        iob1_tags_to_spans,
        bmes_tags_to_spans,
        TypedStringSpan
)


TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]  # pylint: disable=invalid-name


def get_spans_from_arrays(starts: List[int], ends: List[int]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start_idx = 0
    inside = False
    for i in range(len(starts)):
        if starts[i]:
            inside = True
            start_idx = i
            # we ignore double starts without corresponding ends therefore
        if ends[i] and inside:
            spans.append((start_idx, i))
            inside = False
    return spans


@Metric.register("span_ner_as_qa_f1")
class NerAsQaSpanF1(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model performance during training. This metric
    works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O" if using the "BIO" label encoding).

    """
    def __init__(self,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        label_encoding : ``str``, optional (default = "BIO")
            The encoding used to specify label span endpoints in the sequence.
            Valid options are "BIO", "IOB1", "BIOUL" or "BMES".
        tags_to_spans_function: ``Callable``, optional (default = ``None``)
            If ``label_encoding`` is ``None``, ``tags_to_spans_function`` will be
            used to generate spans.
        """
        self._tags_to_spans_function = tags_to_spans_function
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 start_logits: torch.Tensor,
                 end_logits: torch.Tensor,
                 gold_answer_starts: torch.Tensor,
                 gold_answer_ends: torch.Tensor,
                 entity_labels: List[str],
                 ):
        """Note that method signature is really different"""
        if len(entity_labels) != start_logits.size(0):
            raise RuntimeError(f'Number of entity labels {entity_labels} != batch size {start_logits.size(0)}')
        start_logits, end_logits, gold_answer_starts, gold_answer_ends = self.unwrap_to_tensors(
            start_logits,
                                                                                end_logits,
                                                                                gold_answer_starts,
                                                                                gold_answer_ends,
                                                                                )

        for sl, el, gas, gae, entity_label in zip(
                start_logits.argmax(dim=2).tolist(),
                end_logits.argmax(dim=2).tolist(),
                gold_answer_starts.tolist(),
                gold_answer_ends.tolist(),
                entity_labels
        ):
            predicted_spans = get_spans_from_arrays(sl, el)
            gold_spans = set(get_spans_from_arrays(gas, gae))

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[entity_label] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[entity_label] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[entity_label] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
