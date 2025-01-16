import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model,PeftModel, get_peft_model_state_dict
from datasets import load_dataset,Dataset
import argparse
from utils import replicability
import bitsandbytes as bnb
import json
import os
import tqdm

torch.backends.cuda.matmul.allow_tf32 = True # A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs.
IGNORE_INDEX = -100

PROMPTER = {
    "his2query": {
        "template": "Instruction: Based on the following conversation history, please generate a search query that retrieves documents relevant to the next expected utterance.\nConversational history: {history}\nGenerated query:",
        "spliter": "Generated query:"
    },
    "his_cur2query": {
        "template": "Instruction: Based on the following conversation history and the user current utterance, please generate a search query that retrieves documents relevant to the user current utterance.\nConversational history: {history}\nUser current utterance: {current}\nGenerated query:",
        "spliter": "Generated query:"
    }
}

"""
We define two prompts:
 "his2query" aims to generate ad-hoc queries based on only conversational history
"his_cur2query" aims to generate ad-hoc queries based on conversational history as well as the user curren utterance

Note that "spliter" is used for text sparser to extract generated content we want
"""

def load_data(args):

    his = {}
    with open(args.history_dir, 'r') as r:
        for line in r.readlines():
            qid, qtext = line.split('\t')
            his[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    cur = {}
    with open(args.current_dir, 'r') as r:
        for line in r.readlines():
            qid, qtext = line.split('\t')
            cur[qid] = qtext.replace("\t", "").replace("\n", "").replace("\r", "")

    if not args.infer:
        """
        We don't need load query training labels during inference.
        """
        query = {}
        with open(args.query_dir, 'r') as r:
            for line in r.readlines():
                qid_docid_num, qtext = line.split('\t')
                qid = qid_docid_num.split("@")[0]
                if qid not in query:
                    query[qid] = []
                query[qid].append(qtext.replace("\t", "").replace("\n", "").replace("\r", ""))

    examples = []

    for qid in his.keys():
        example = {}
        example["example_id"] = qid

        # adding input
        if args.prompt == "his2query":
            example["input"] = PROMPTER[args.prompt]["template"].format(history=his[qid])
        elif args.prompt == "his_cur2query":
            example["input"] = PROMPTER[args.prompt]["template"].format(history=his[qid], current=cur[qid])
        else:
            raise NotImplemented

        # adding learning targets during training
        if not args.infer:
            """
            We don't need load query training labels during inference.
            """
            example["output"] = " | ".join(query[qid])

        examples.append(example)

    if args.verbose:
        print(f"# examples {len(examples)}")
        print(f"{examples[0]}\n\n{examples[1]}\n\n{examples[2]}\n\n{examples[3]}\n\n{examples[-1]}")

    return examples

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train(args):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    training_args = transformers.TrainingArguments(
        # remove_unused_columns=False, #  Whether or not to automatically remove the columns unused by the model forward method
        local_rank=args.local_rank,
        report_to='wandb',  # default to ['tensorboard', 'wandb']
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # warmup_ratio=0.05,
        # max_steps=100,
        save_steps = args.save_steps,
        save_strategy= "steps", #"epoch",
        save_total_limit=None, # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.logging_steps,
        output_dir=args.checkpoint_path,
        optim=args.optim,
        lr_scheduler_type="constant",
        group_by_length=args.group_by_length, # Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.
        deepspeed=args.deepspeed_config,  # Include the DeepSpeed configuration
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        #device_map="auto", #device_map, # we don't need this if we use deepspeed
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        token=args.token,
        quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_has_fp16_weight=False,
    ))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=args.padding_side, cache_dir=args.cache_dir, token=args.token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model.config.torch_dtype =torch.bfloat16
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if args.verbose:
        print(f"model.config:\n{model.config}")
        print(f"model.generation_config:\n{model.generation_config}")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()  # reduce the memeory, but increase the training time

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = find_all_linear_names(model)
    ))

    if args.verbose:
        print_trainable_parameters(model)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True, # Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
            max_length=args.max_input_length,
            padding=False,
            return_tensors=None,
        )

        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.max_input_length
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        prompt_label = " ".join([data_point["input"], data_point["output"]])
        tokenized_prompt_label = tokenize(prompt_label)

        if not args.train_on_inputs:
            prompt = data_point["input"]
            tokenized_prompt = tokenize(prompt, add_eos_token=False)
            prompt_len = len(tokenized_prompt["input_ids"])
            tokenized_prompt_label["labels"] = [IGNORE_INDEX] * prompt_len + tokenized_prompt_label["labels"][prompt_len:]  # could be sped up, probably

        return tokenized_prompt_label

    examples = load_data(args)
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(generate_and_tokenize_prompt)

    if args.verbose:
        print(f"training_args:\n{training_args}")

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train()
    model.save_pretrained(args.checkpoint_path)


def infer(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        token=args.token,
        quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    )

    if args.checkpoint_name:
        model = PeftModel.from_pretrained(model, args.checkpoint_path)
        #model = model.merge_and_unload() # not necessary

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,padding_side=args.padding_side, cache_dir=args.cache_dir, token=args.token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model.config.torch_dtype =torch.bfloat16
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if isinstance(model.generation_config.eos_token_id, list):
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] # llama 3 128001
    else:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id # llama 3 128001

    model.eval()

    examples = load_data(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    it = range(0, len(examples), args.batch_size)

    for start_idx in tqdm.tqdm(it):
        # one batch
        rng = slice(start_idx, start_idx + args.batch_size)

        # padding=True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
        enc = tokenizer([example['input'] for example in examples[rng]], padding=True, truncation=True, max_length=args.max_input_length, return_tensors='pt')

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            predictions = model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                max_new_tokens=args.max_new_tokens,
            )

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for idx, example in enumerate(examples[rng]):
            # text parsing
            prediction = predictions[idx].split(PROMPTER[args.prompt]["spliter"])[-1].strip().replace("\t", "").replace("\n", "").replace("\r", "")
            example["prediction"] = prediction

    with open(f"{args.output_path}.tsv", "w") as w1, open(f"{args.output_path}.jsonl", "w") as w2:
        for idx, example in enumerate(examples):
            qid = example["example_id"]
            text = example["prediction"]

            w1.write(f"{qid}\t{text}\n")
            w2.write(json.dumps({"query_id": qid, "query": text}) + "\n")

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--verbose", action='store_true')

    parser.add_argument("--infer", action='store_true')

    parser.add_argument("--deepspeed_config", type=str, default=None)

    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--history_dir", type=str, default=None)
    parser.add_argument("--current_dir", type=str, default=None)
    parser.add_argument("--query_dir", type=str, default=None)

    parser.add_argument("--prompt", type=str)

    parser.add_argument("--token", type=str)
    parser.add_argument("--cache_dir", type=str)

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--train_on_inputs", action='store_true')
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--truncation_side", type=str, default='left')
    parser.add_argument("--padding_side", type=str, default='left')
    parser.add_argument("--max_input_length", type=int, default=2048) # 2048
    parser.add_argument("--max_new_tokens", type=int, default=128) # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)  # 1e-4
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--group_by_length", action='store_true')
    parser.add_argument("--num_epochs", type=float, default=5.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=float, default=1000)

    parser.add_argument("--lora_r", type=int, default=64)  # [64, 16, 8]  # 256？
    parser.add_argument("--lora_alpha", type=int, default=16) # [32, 16]
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    args = parser.parse_args()

    # parse basic information from input path
    args.dataset_class = args.history_dir.split("/")[-1].split(".")[0]
    args.dataset_name = args.history_dir.split("/")[-1].split(".")[1]
    args.history_type = args.history_dir.split("/")[-1].split(".")[3]
    args.base_model = args.model_name_or_path.split("/")[-1]

    if args.verbose:
        print(f"dataset_class: {args.dataset_class}\ndataset_name: {args.dataset_name}\nhistory_type: {args.history_type}\nbase_model: {args.base_model}\n")

    # make sure replicability
    replicability(seed=args.random_seed)

    if args.infer is True:
        # when we do inference
        if args.checkpoint_name:
            # when we use a fine-tuned checkpoint
            args.checkpoint_path = f"{args.checkpoint_dir}/{args.checkpoint_name}/" # a specific checkpoint file name
            if "/" in args.checkpoint_name:
                args.checkpoint_name = args.checkpoint_name.replace("/", "-")
            args.setup = f"{args.dataset_class}.{args.dataset_name}.q.{args.prompt}--{args.base_model}--ckpt-{args.checkpoint_name}"
        else:
            # when we do infernece based on an original model
            args.setup = f"{args.dataset_class}.{args.dataset_name}.q.{args.prompt}--{args.base_model}"

        # define where output files will be generated
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # a specific output file name
        args.output_path = f"{args.output_dir}/{args.setup}"

        infer(args)
    else:
        # when we do fine-tuning

        # extract learning target name
        args.output_type = args.query_dir.split("/")[-1].split(".")[3]
        # define checkpoint name
        args.setup = f"{args.dataset_class}.{args.dataset_name}.q.{args.prompt}--{args.base_model}--{args.output_type}"
        # define where checkpoint will be saved
        args.checkpoint_path = f"{args.checkpoint_dir}/{args.setup}/"

        train(args)