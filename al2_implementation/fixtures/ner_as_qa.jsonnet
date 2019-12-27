local embedding_dim = 16;
local data_path = "./fixtures/ontonotes_v3_example.conll";
{
  "dataset_reader": {
    "type": "ner_qa",
    "descriptions_path": "./fixtures/ontonotes_descriptions.json",
    "descriptions_type": "natural_query",
    "target_column": 3
  },
  "train_data_path": data_path,
  "validation_data_path": data_path,
  "model": {
    "type": "ner_as_mrc",
    "word_embeddings": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embedding_dim,
        "trainable": true
      }
    },
    "encoder": {
        "type": "lstm",
        "input_size": embedding_dim,
        "hidden_size": 50,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": true
    },
  },
  "iterator": {
    "type": "basic",

  },
  "trainer": {
    "cuda_device": -1,
    "grad_norm": 5.0,
    "validation_metric": "+f1-measure",
    "shuffle": false,
    "optimizer": {
      "type": "adam",
      "lr": 0.005
    },
    "num_epochs": 200,
    "patience": 200,
  }
}
