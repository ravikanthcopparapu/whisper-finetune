import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models.')

parser.add_argument('--model_name', type=str, default='openai/whisper-small')
parser.add_argument('--language', type=str, default='Hindi')
parser.add_argument('--sampling_rate', type=int, default=16000)
parser.add_argument('--num_proc', type=int, default=2)
parser.add_argument('--train_strategy', type=str, default='epoch')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--warmup', type=int, default=500)
parser.add_argument('--train_batchsize', type=int, default=2)
parser.add_argument('--eval_batchsize', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--resume_from_ckpt', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='op_dir_epoch')
parser.add_argument('--train_datasets', type=str, nargs='+', required=True)
parser.add_argument('--eval_datasets', type=str, nargs='+', required=True)

args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('train_strategy must be either "steps" or "epoch"')

print('\nARGS:')
print(vars(args))
print('--------------------------------------------')

#############################       CONFIG FLAGS       #####################################

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()

#############################       MODEL LOADING       #####################################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

if model.config.decoder_start_token_id is None:
    raise ValueError("decoder_start_token_id is not defined")

# Enable gradient checkpointing for low memory (Colab)
if gradient_checkpointing:
    model.gradient_checkpointing_enable()

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False

# IMPORTANT: Do NOT disable language conditioning
# model.config.forced_decoder_ids = None
# model.config.suppress_tokens = []

############################        DATASET LOADING AND PREP        ##########################

def load_custom_dataset(split):
    ds = []
    if split == 'train':
        for dset in args.train_datasets:
            ds.append(load_from_disk(dset))
    if split == 'eval':
        for dset in args.eval_datasets:
            ds.append(load_from_disk(dset))

    ds_to_return = concatenate_datasets(ds)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return

def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()

    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 20.0

def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

print('Preparing datasets...')

raw_dataset = DatasetDict()
raw_dataset["train"] = load_custom_dataset('train')
raw_dataset["eval"] = load_custom_dataset('eval')

raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=args.num_proc,
)

print('Dataset preparation done.')

###############################     DATA COLLATOR     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

###############################     METRICS     ########################

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(p) for p in pred_str]
        label_str = [normalizer(l) for l in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

###############################     TRAINING ARGS     ########################

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_torch",
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_torch",
    )

###############################     TRAINER     ########################

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

###############################     TRAIN     ########################

print('TRAINING IN PROGRESS...')
trainer.train()
print('DONE TRAINING')
