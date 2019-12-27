# pylint: disable=invalid-name,protected-access
import unittest
from copy import deepcopy

from flaky import flaky
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from al2_implementation.ner_qa_model import NerAsQaModel
from al2_implementation.ner_qa_reader import NerAsReadingComprehensionDatasetReader


class SimpleRETest(ModelTestCase):
    # FIXTURES_ROOT = PROJECT_ROOT / "components" / "tests" / "fixtures"

    def setUp(self):
        super().setUp()
        self.set_up_model(
            "fixtures/ner_as_qa.jsonnet",
            "./fixtures/ontonotes_v3_example.conll",
        )

    def test_01_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    @pytest.mark.skip
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        logits = output_dict["logits"]
        assert logits.shape[-1] == 2
        # assert len(logits[0]) == 7
        # assert len(logits[1]) == 7
        # for example_tags in logits:
        #     for tag_id in example_tags:
        #         tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
        #         assert tag in {'0', '1'}

