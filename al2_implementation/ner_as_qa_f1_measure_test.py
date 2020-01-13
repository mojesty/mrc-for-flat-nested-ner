# pylint: disable=no-self-use,invalid-name,protected-access
import os
import subprocess

import torch
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.training.metrics import SpanBasedF1Measure, Metric
from allennlp.models.semantic_role_labeler import write_bio_formatted_tags_to_file
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError

from al2_implementation.ner_as_qa_f1_measure import NerAsQaSpanF1


class SpanBasedF1Test(AllenNlpTestCase):

    def setUp(self):
        super(SpanBasedF1Test, self).setUp()
        vocab = Vocabulary()
        vocab.add_token_to_namespace("O", "tags")
        vocab.add_token_to_namespace("B-ARG1", "tags")
        vocab.add_token_to_namespace("I-ARG1", "tags")
        vocab.add_token_to_namespace("B-ARG2", "tags")
        vocab.add_token_to_namespace("I-ARG2", "tags")
        vocab.add_token_to_namespace("B-V", "tags")
        vocab.add_token_to_namespace("I-V", "tags")
        vocab.add_token_to_namespace("U-ARG1", "tags")
        vocab.add_token_to_namespace("U-ARG2", "tags")
        vocab.add_token_to_namespace("B-C-ARG1", "tags")
        vocab.add_token_to_namespace("I-C-ARG1", "tags")
        vocab.add_token_to_namespace("B-ARGM-ADJ", "tags")
        vocab.add_token_to_namespace("I-ARGM-ADJ", "tags")

        # BMES.
        vocab.add_token_to_namespace("B", "bmes_tags")
        vocab.add_token_to_namespace("M", "bmes_tags")
        vocab.add_token_to_namespace("E", "bmes_tags")
        vocab.add_token_to_namespace("S", "bmes_tags")

        self.vocab = vocab

    def test_01_span_metrics_are_computed_correcly_in_simplest_case(self):
        # note that logits are unnormalized. That's OK because
        # we compute argmax over logits anyways
        metric = NerAsQaSpanF1(self.vocab)
        metric(torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ]]),
               torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.7, .3],
                              [.6, .4],
                              [.6, .5],
                              [.1, .9]
                              ]]),
               torch.LongTensor([[0, 0, 0, 0, 1, 0]]),
               torch.LongTensor([[0, 0, 0, 0, 0, 1]]),
               ['test1']
               )
        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0, decimal=3)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 1.0)

    def test_02_span_metrics_are_computed_correcly_in_one_token_entity(self):
        # note that logits are unnormalized. That's OK because
        # we compute argmax over logits anyways
        metric = NerAsQaSpanF1(self.vocab)
        metric(torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ]]),
               torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.7, .3],
                              [.6, .4],
                              [.2, .8],
                              [.1, .9]
                              ]]),
               torch.LongTensor([[0, 0, 0, 0, 1, 0]]),
               torch.LongTensor([[0, 0, 0, 0, 1, 0]]),
               ['test1']
               )
        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0, decimal=3)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 1.0)

    def test_03_span_metrics_are_computed_correcly_in_double_case(self):
        # note that logits are unnormalized. That's OK because
        # we compute argmax over logits anyways
        metric = NerAsQaSpanF1(self.vocab)
        metric(torch.Tensor([[[.7, .3],
                              [.2, .8],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ]]),
               torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.2, .8],
                              [.6, .4],
                              [.6, .5],
                              [.1, .9]
                              ]]),
               torch.LongTensor([[0, 1, 0, 0, 1, 0]]),
               torch.LongTensor([[0, 0, 1, 0, 0, 1]]),
               ['test1']
               )
        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0, decimal=3)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 1.0)

    def test_04_span_metrics_are_computed_correcly_in_mixed_case(self):
        # note that logits are unnormalized. That's OK because
        # we compute argmax over logits anyways
        metric = NerAsQaSpanF1(self.vocab)
        metric(torch.Tensor([[[.7, .3],
                              [.2, .8],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ]]),
               torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.55, .5],
                              [.6, .4],
                              [.6, .5],
                              [.1, .9]
                              ]]),
               torch.LongTensor([[0, 1, 0, 0, 1, 0]]),
               torch.LongTensor([[0, 0, 1, 0, 0, 1]]),
               ['test1']
               )
        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0, decimal=3)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.6667, decimal=3)

    def test_05_span_metrics_are_computed_correcly_in_fail_case(self):
        # note that logits are unnormalized. That's OK because
        # we compute argmax over logits anyways
        metric = NerAsQaSpanF1(self.vocab)
        metric(torch.Tensor([[[.7, .3],
                              [.2, .8],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ]]),
               torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.55, .5],
                              [.6, .4],
                              [.6, .5],
                              [.6, .5]
                              ]]),
               torch.LongTensor([[0, 1, 0, 0, 1, 0]]),
               torch.LongTensor([[0, 0, 1, 0, 0, 1]]),
               ['test1']
               )
        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 0.0, decimal=3)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.0, decimal=3)

    def test_06_span_metrics_are_computed_correcly_for_multiple_labels(self):
        # note that logits are unnormalized. That's OK because
        # we compute argmax over logits anyways
        metric = NerAsQaSpanF1(self.vocab)
        metric(torch.Tensor([[[.7, .3],
                              [.2, .8],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ],
                             [[.7, .3],
                              [.2, .8],
                              [.7, .3],
                              [.6, .4],
                              [.1, .9],
                              [.6, .5]
                              ]
                             ]),
               torch.Tensor([[[.7, .3],
                              [.6, .4],
                              [.55, .5],
                              [.6, .4],
                              [.6, .5],
                              [.1, .9]
                              ],
                             [[.7, .3],
                              [.6, .4],
                              [.55, .5],
                              [.6, .4],
                              [.6, .5],
                              [.1, .9]
                              ]
                             ]),
               torch.LongTensor([[0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]]),
               torch.LongTensor([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]),
               ['test1', 'test2']
               )
        metric_dict = metric.get_metric()
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0, decimal=3)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.6667, decimal=3)

    def test_span_f1_can_build_from_params(self):
        params = Params({"type": "span_ner_as_qa_f1"})
        metric = Metric.from_params(params=params, vocabulary=self.vocab)
