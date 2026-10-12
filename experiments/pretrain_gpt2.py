import os
import argparse
from datetime import datetime
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from utils.data import load_limited_babylm

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="gpt2_small")
parser.add_argument("--out_root", type=str, default="./models")

parser.add_argument("--vocab_size", type=int, default=16384)
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--dataset_limit_words", type=int, default=9_000_000)

CHECKPOINT_STEPS = {
    *(186 * i for i in range(1,10)),      # 1M–9M
    *(1860 * i for i in range(1, 10)),     # 10M–90M
    *(18600 * i for i in range(1, 11)),     # 100M–1B
}

class SaveAtStepsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in CHECKPOINT_STEPS:
            control.should_save = True
        return control

def main(args):
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(args.out_root, args.run_id)
    model_dir = run_dir
    tokenizer_dir = os.path.join(run_dir, "tokenizer")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    dataset = load_limited_babylm(max_words=args.dataset_limit_words)

    text_column = "text"
    if text_column not in dataset.column_names:
        text_column = dataset.column_names[0]

    texts = dataset[text_column]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>"],
    )

    tokenizer.save_model(tokenizer_dir)

    hf_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.bos_token = "<bos>"
    hf_tokenizer.eos_token = "<eos>"

    def tokenize(example):
        return hf_tokenizer(example[text_column])

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concat["input_ids"])
        total_len = (total_len // args.block_size) * args.block_size

        result = {
            k: [t[i:i + args.block_size] for i in range(0, total_len, args.block_size)]
            for k, t in concat.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)

    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=384,
        n_layer=12,
        n_head=6,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        bos_token_id=1,
        eos_token_id=2,
    )

    model = GPT2LMHeadModel(config)
    if torch.cuda.is_available():
        model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        warmup_steps=800,
        lr_scheduler_type="cosine",
        seed=42,
        logging_steps=50,
        save_steps=10000,
        fp16=True,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=hf_tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=collator,
        callbacks=[SaveAtStepsCallback()],
    )

    trainer.train()

    model.save_pretrained(model_dir)
    hf_tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)