import unittest

from allennlp.common.util import ensure_list
from allennlp.data import Token

from al2_implementation.ner_qa_reader import NerAsReadingComprehensionDatasetReader


class TestNerQAReader(unittest.TestCase):

    def test_01_read_file(self):
        reader = NerAsReadingComprehensionDatasetReader(
            descriptions_path='fixtures/ontonotes_descriptions.json',
            descriptions_type='natural_query',
            target_column=3
        )

        instances = reader.read('fixtures/ontonotes_v3_example.conll')
        instances = ensure_list(instances)
        self.assertEqual(len(instances), 4)
        # checking 1st instance
        inst = instances[0]
        self.assertEqual(sum(inst['answer_starts']), 1)
        self.assertIn(Token('[SEP]'), inst['context'].tokens, 'context is not well-formed "[CLS] query [SEP] text" question')
        self.assertEqual(inst['context'].tokens.index(Token('nationalities')), 1)
        self.assertEqual(inst['answer_starts'], inst['answer_ends'])
        self.assertEqual(inst['answer_starts'].labels.index(True), 14)
        self.assertEqual(inst['meta'].metadata['type'], 'NORP')
        # 3rd and 4th instances must have same context but different queries
        self.assertEqual(instances[2]['meta'].metadata['text'], instances[3]['meta'].metadata['text'])
        self.assertNotEqual(instances[2]['meta'].metadata['query'], instances[3]['meta'].metadata['query'])


if __name__ == '__main__':
    unittest.main()
