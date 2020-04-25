import math

import torch
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer

BERT_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking'
)


class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


def get_model_and_tokenizer(args):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=bool('uncased' in args.model_name_or_path),
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    model = BertWSD.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # add new special token
    if '[TGT]' not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        assert '[TGT]' in tokenizer.additional_special_tokens
        model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return model, tokenizer


def _forward(args, model, batch):
    batch = tuple(t.to(args.device) for t in batch)
    outputs = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])

    return model.dropout(outputs[1])


def _compute_weighted_loss(loss, weighting_factor):
    squared_factor = weighting_factor ** 2

    return 1 / (2 * squared_factor) * loss + math.log(1 + squared_factor)


def forward_gloss_selection(args, model, batches):
    batch_loss = 0
    logits_list = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in batches:
        logits = model.ranking_linear(_forward(args, model, batch)).squeeze(-1)
        labels = torch.max(batch[3].to(args.device), -1).indices.to(args.device).detach()

        batch_loss += loss_fn(logits.unsqueeze(dim=0), labels.unsqueeze(dim=-1))
        logits_list.append(logits)

    loss = batch_loss / len(batches)

    return loss, logits_list

