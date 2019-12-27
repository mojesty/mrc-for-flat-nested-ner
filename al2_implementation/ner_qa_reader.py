import itertools
import json
from pathlib import Path
from typing import Iterable, List, Dict

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan
from allennlp.data.dataset_readers.dataset_utils.span_utils import iob1_tags_to_spans
from allennlp.data.fields import TextField, IndexField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


@DatasetReader.register('ner_qa')
class NerAsReadingComprehensionDatasetReader(DatasetReader):

    def __init__(self,
                 descriptions_path: str,
                 descriptions_type: str,
                 target_column: int = 1,
                 skip_empty: bool = False,
                 classes_to_ignore: List[str] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True):
        super(NerAsReadingComprehensionDatasetReader, self).__init__(lazy=lazy)
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._descriptions = json.loads(Path(descriptions_path).read_text())
        if descriptions_type not in self._descriptions:
            raise RuntimeError(f'Context type {descriptions_type} not found from descriptions in {descriptions_path}')
        self._context_type = descriptions_type
        self._skip_empty = skip_empty
        self._target_column = target_column

    def _read(self, file_path: str) -> Iterable[Instance]:
        sentences = Path(file_path).read_text().split('\n\n')
        for sentence in sentences:
            tokens = []
            targets = []
            for token_line in sentence.split('\n'):
                text, *labels = token_line.split('\t')
                tokens.append(text)
                targets.append(labels[self._target_column - 1])
            entities_spans: List[TypedStringSpan] = iob1_tags_to_spans(targets)
            # one sentence may have several entities of the same type, so it is not
            # exactly squad-like task, where only 1 span may be answer
            # to take this into account, we group spans by their type
            # and for each _group_ create an instance with _multiple_
            # answer spans. The model learns to predict start and end token
            # for _each_ of these spans.
            entities_spans_grouped = itertools.groupby(sorted(entities_spans, key=lambda x: x[0]), key=lambda x: x[0])
            for key, group in entities_spans_grouped:
                spans = list(group)
                query = self._descriptions[self._context_type][key]
                query = query.split(' ')  # TODO (yaroslav): make this better 27.12.19
                answer_starts: List[int] = [span[1][0] for span in spans]
                answer_ends: List[int] = [span[1][1] for span in spans]
                yield self.text_to_instance(tokens, query, answer_starts, answer_ends, type=key)

    def text_to_instance(self,
                         tokens: List[str],
                         query: List[str],
                         answer_starts: List[int] = None,
                         answer_ends: List[int] = None,
                         **metadata
                         ) -> Instance:
        if (answer_starts is not None) ^ (answer_ends is not None):
            raise RuntimeError(f'Answer starts and answer ends must be or be not simultaneously')
        # text_field = TextField([Token(word) for word in tokens], self._token_indexers)
        # query_field = TextField([Token(word) for word in query], self._token_indexers)
        text_and_query_field = TextField(
            [Token('[CLS]')] + [Token(word) for word in query]
            + [Token('[SEP]')] + [Token(word) for word in tokens],
            self._token_indexers
        )
        offset = len(query) + 2
        metadata.update({'text': tokens, 'query': query})
        fields = {
            'context': text_and_query_field,
            'meta': MetadataField(metadata)
        }
        answer_starts = set(answer_starts)
        answer_ends = set(answer_ends)
        # fields['answer_starts'] = IndexField(span_start, passage_field)
        if answer_starts is not None:
            fields['answer_starts'] = SequenceLabelField(
                [False] * offset + [i in answer_starts for i in range(len(tokens))],
                sequence_field=text_and_query_field
            )
            fields['answer_ends'] = SequenceLabelField(
                [False] * offset + [i in answer_ends for i in range(len(tokens))],
                sequence_field=text_and_query_field
            )
        return Instance(fields)
