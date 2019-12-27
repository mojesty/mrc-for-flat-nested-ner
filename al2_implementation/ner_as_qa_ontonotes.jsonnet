local bertname = 'bert-base-cased';
local data_prefix = std.extVar('ONTONOTES_PATH');
{
  "dataset_reader": {
    "type": "ner_qa",
    "descriptions_path": "al2_implementation/fixtures/ontonotes_descriptions.json",
    "descriptions_type": "natural_query",
    "target_column": 3,
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bertname,
        "use_starting_offsets": true,
        "do_lowercase": false
      },
    }
  },
  "train_data_path": data_prefix + "/train.txt",
  "validation_data_path": data_prefix + "/valid.txt",
  //"test_data_path": data_prefix + "/test.json",
  "model": {
    "type": "ner_as_mrc",
    "word_embeddings": {
      "token_embedders": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": bertname,
            "top_layer_only": false,
            "requires_grad": true,
          },
        },
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets"]
        },
        "allow_unmatched_keys": true
    },
    "encoder": {
        "type": "pass_through",
        "input_dim": 768,
    },
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 4,
    "sorting_keys": [["context", "num_tokens"]],
    "biggest_batch_first": true,
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.0005
    },
    "validation_metric": "+f1-measure",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0
  }
}
