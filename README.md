# BERT-WSD
This is the official code repository for the Findings of EMNLP 2020 paper "[Adapting BERT for Word Sense Disambiguation with Gloss Selection Objective and Example Sentences](https://arxiv.org/abs/2009.11795)".
 
## Installation
```
python3 -m virtualenv env
source env/bin/activate

pip install -r requirements.txt
```


## Pre-trained models
All datasets and pre-trained models are available for download [here](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EpCcVDhHWtZKp50duiIcGbABIpPfUhok-vJisRk7Ri9RnA?e=cHTrUp). 

### Experiment results
| Checkpoint           | Parameters    | SE07      | SE2      | SE3      | SE13     | SE15     | ALL      |
| ---------------------|:-------------:|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [BERT-base-baseline](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EqmcCr9jiCJFt0uQ8WhjmQwBZL3skr1b-J01NnNo7NEJPg?e=MAmNSB)   | 110M          | **73.6**  | 79.4     | 76.8     | 77.4     | 81.5     | 78.2     |
| [BERT-base-augmented](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EiWzblOyyOBDtuO3klUbXoAB3THFzke-2MLWguIXrDopWg?e=08umXD)  | 110M          | **73.6**  | 79.3     | 76.9     | 79.1     | 82.0     | 78.7     | 
| [BERT-large-baseline](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/Ep1Uw0RBthJJv-pGAJtmOiQBFIXB3fAXuGYDxxNRRLrlbg?e=YgSa1T)  | 340M          | 73.0      | **79.9** | 77.4     | 78.2     | 81.8     | 78.7     | 
| [BERT-large-augmented](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EqZjlCC79rRKrEUWBEm6s98BzeQYMWZNydAKLOzGDgD8Eg?e=rjZKTV) | 340M          | 72.7      | 79.8     | **77.8** | **79.7** | **84.4** | **79.5** |

### Command line demo
Usage:
```
python script/demo_model.py model_dir

positional arguments:
  model_dir   Directory of pre-trained model.
```

Example:
```
>> python script/demo_model.py "model/bert_large-augmented-batch_size=128-lr=2e-5-max_gloss=6"
Loading model...

Enter a sentence with an ambiguous word surrounded by [TGT] tokens
> He caught a [TGT] bass [TGT] yesterday.
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:01<00:00,  5.46it/s]

Predictions:
  No.  Sense key            Definition                                                                                             Score
-----  -------------------  ---------------------------------------------------------------------------------------------------  -------
    1  bass%1:13:02::       the lean flesh of a saltwater fish of the family Serranidae                                          0.53924
    2  bass%1:13:01::       any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)  0.3907
    3  bass%1:05:00::       nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes               0.05478
    4  bass%5:00:00:low:03  having or denoting a low vocal or instrumental range                                                 0.0046
    5  bass%1:10:00::       the lowest adult male singing voice                                                                  0.00361
    6  bass%1:18:00::       an adult male singer with the lowest voice                                                           0.00318
    7  bass%1:06:02::       the member with the lowest range of a family of musical instruments                                  0.00226
    8  bass%1:07:01::       the lowest part of the musical range                                                                 0.00136
    9  bass%1:10:01::       the lowest part in polyphonic music                                                                  0.00028

Enter a sentence with an ambiguous word surrounded by [TGT] tokens
> It's all about that [TGT] bass [TGT].
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:01<00:00,  7.00it/s]

Predictions:
  No.  Sense key            Definition                                                                                             Score
-----  -------------------  ---------------------------------------------------------------------------------------------------  -------
    1  bass%1:10:01::       the lowest part in polyphonic music                                                                  0.27318
    2  bass%1:10:00::       the lowest adult male singing voice                                                                  0.2048
    3  bass%5:00:00:low:03  having or denoting a low vocal or instrumental range                                                 0.19921
    4  bass%1:06:02::       the member with the lowest range of a family of musical instruments                                  0.18751
    5  bass%1:07:01::       the lowest part of the musical range                                                                 0.11289
    6  bass%1:18:00::       an adult male singer with the lowest voice                                                           0.009
    7  bass%1:13:02::       the lean flesh of a saltwater fish of the family Serranidae                                          0.00776
    8  bass%1:13:01::       any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)  0.0035
    9  bass%1:05:00::       nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes               0.00215
```


## Dataset preparation
### Training dataset
Usage:
```
python script/prepare_dataset.py --corpus_dir CORPUS_DIR --output_dir OUTPUT_DIR
                                 --max_num_gloss MAX_NUM_GLOSS
                                 [--use_augmentation]

arguments:
  --corpus_dir CORPUS_DIR
                        Path to directory consisting of a .xml file and a .txt
                        file corresponding to the sense-annotated data and its
                        gold keys respectively.
  --output_dir OUTPUT_DIR
                        The output directory where the .csv file will be
                        written.
  --max_num_gloss MAX_NUM_GLOSS
                        Maximum number of candidate glosses a record can have
                        (include glosses from ground truths)
  --use_augmentation    Whether to augment training dataset with example
                        sentences from WordNet
```

Example:
```
python script/prepare_dataset.py \
    --corpus_dir "data/corpus/SemCor" \
    --output_dir "data/train" \
    --max_num_gloss 6 \
    --use_augmentation
```

### Development/Test dataset
Usage:
```
python script/prepare_dataset.py --corpus_dir CORPUS_DIR --output_dir OUTPUT_DIR

arguments:
  --corpus_dir CORPUS_DIR
                        Path to directory consisting of a .xml file and a .txt
                        file corresponding to the sense-annotated data and its
                        gold keys respectively.
  --output_dir OUTPUT_DIR
                        The output directory where the .csv file will be
                        written.
```

Example:
```
python script/prepare_dataset.py \
    --corpus_dir "data/corpus/semeval2007" \
    --output_dir "data/dev"
```


## Fine-tuning BERT
Usage:
```
python script/run_model.py --do_train --train_path TRAIN_PATH
                           --model_name_or_path MODEL_NAME_OR_PATH 
                           --output_dir OUTPUT_DIR
                           [--evaluate_during_training]
                           [--eval_path EVAL_PATH]
                           [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                           [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                           [--learning_rate LEARNING_RATE]
                           [--num_train_epochs NUM_TRAIN_EPOCHS]
                           [--logging_steps LOGGING_STEPS] 
                           [--save_steps SAVE_STEPS]

arguments:
  --do_train            Whether to run training on train set.
  --train_path TRAIN_PATH
                        Path to training dataset (.csv file).
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or shortcut name selected in
                        the list: bert-base-uncased, bert-large-uncased, bert-
                        base-cased, bert-large-cased, bert-large-uncased-
                        whole-word-masking, bert-large-cased-whole-word-
                        masking
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --evaluate_during_training
                        Run evaluation during training at each logging step.
  --eval_path EVAL_PATH
                        Path to evaluation dataset (.csv file).
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
```

Example:
```
python script/run_model.py \
    --do_train \
    --train_path "data/train/semcor-max_num_gloss=6-augmented.csv" \
    --model_name_or_path "bert-base-uncased" \
    --output_dir "model/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6" \
    --evaluate_during_training \
    --eval_path "data/dev/semeval2007.csv" \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --logging_steps 1000 \
    --save_steps 1000
```


## Evaluation
### Generate predictions
Usage:
```
python script/run_model.py --do_eval --eval_path EVAL_PATH
                           --model_name_or_path MODEL_NAME_OR_PATH 
                           --output_dir OUTPUT_DIR

arguments:
  --do_eval             Whether to run evaluation on dev/test set.
  --eval_path EVAL_PATH
                        Path to evaluation dataset (.csv file).
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
```
Example:
```
python script/run_model.py \
    --do_eval \
    --eval_path "data/dev/semeval2007.csv" \
    --model_name_or_path "model/bert_large-augmented-batch_size=128-lr=2e-5-max_gloss=6" \
    --output_dir "model/bert_large-augmented-batch_size=128-lr=2e-5-max_gloss=6"
```

### Scoring
Usage:
```
java Scorer GOLD_KEYS PREDICTIONS

arguments:
  GOLD_KEYS    Path to gold key file
  PREDICTIONS  Path to predictions file
```
Example:
```
java Scorer data/corpus/semeval2007/semeval2007.gold.key.txt \
    model/bert_large-augmented-batch_size=128-lr=2e-5-max_gloss=6/semeval2007_predictions.txt
```


## References
- Raganato, Alessandro, Jose Camacho-Collados, and Roberto Navigli. "Word sense disambiguation: A unified evaluation framework and empirical comparison." Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers. 2017.
- Huang, Luyao, et al. "GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.
- Wolf, Thomas, et al. "Huggingface’s transformers: State-of-the-art natural language processing." ArXiv, abs/1910.03771 (2019).


## Citation
```
@inproceedings{yap-etal-2020-adapting,
    title = "Adapting {BERT} for Word Sense Disambiguation with Gloss Selection Objective and Example Sentences",
    author = "Yap, Boon Peng  and
      Koh, Andrew  and
      Chng, Eng Siong",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.4",
    pages = "41--46"
}
```


