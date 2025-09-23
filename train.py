import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import json

import torch
import torch.nn.functional as F
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": "Attention implementation to use. E.g., 'flash_attention_2' or 'sdpa'."})
    full_finetune: Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(default=None, metadata={"help": ("Pre-initialized PiSSA adapter path."),},)
    init_weights: bool | str = field(default=True, metadata={"help": ("True -> LoRA; `pissa` -> PiSSA."),},)
    use_dora: Optional[bool] = field(default=False)
    target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=32.)
    lora_dropout: Optional[float] = field(default=0., metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset: Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."},)
    merge: Optional[bool] = field(default=False, metadata={"help": "Merge the adapter to the model."},)

    # BA-LoRA (Bias-Alleviating Low-Rank Adaptation) Arguments
    use_ba_lora: bool = field(default=False, metadata={"help": "Whether to use BA-LoRA regularization."})
    base_model_for_pt: Optional[str] = field(default=None, metadata={"help": "Path for consistency regularization."})
    lambda1: float = field(default=0.0, metadata={"help": "Base weight for L_CR_NLG."})
    lambda2: float = field(default=0.0, metadata={"help": "Weight for L_DR_NLG."})
    lambda3: float = field(default=0.0, metadata={"help": "Weight for L_SVDR_NLG."})
    svd_k: int = field(default=1, metadata={"help": "Base rank for SVD regularization."})
    top_k_entropy: int = field(default=0, metadata={"help": "If > 0, use Top-K entropy for diversity."})
    distill_temp: float = field(default=1.0, metadata={"help": "Base temperature for KL loss."})
    svd_frob_norm: bool = field(default=False, metadata={"help": "Use Frobenius norm for SVD loss."})
    lambda1_schedule: Optional[str] = field(default=None, metadata={"help": "Scheduling for lambda1. e.g., 'cosine'."})
    lambda_focus_schedule: Optional[str] = field(default=None, metadata={"help": "Scheduling for lambda2/3. e.g., 'linear_warmup', 'two_phase', or 'quadratic_warmup'."})
    lambda_warmup_ratio: float = field(default=0.1, metadata={"help": "Duration ratio for the focus schedule."})
    lambda_ramp_up_ratio: float = field(default=0.05, metadata={"help": "In 'two_phase', defines the ramp-up duration."})
    use_adaptive_regularization: bool = field(default=False, metadata={"help": "Enable data-driven adaptive regularization for lambda1."})
    lambda1_min: Optional[float] = field(default=None, metadata={"help": "The minimum value for adaptive lambda1."})
    lambda1_max: Optional[float] = field(default=None, metadata={"help": "The maximum value for adaptive lambda1."})


class BALoRATrainer(Trainer):
    def __init__(self, pt_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.use_ba_lora:
            if pt_model is None:
                raise ValueError("pt_model is required for BA-LoRA.")
            self.pt_model = pt_model
            self.pt_model.eval()
            self.pt_model.requires_grad_(False)
            if self.args.use_adaptive_regularization:
                self.max_entropy = np.log(self.model.config.vocab_size)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        task_loss = outputs.loss
        if not self.args.use_ba_lora:
            return (task_loss, outputs) if return_outputs else task_loss

        ft_logits = outputs.logits
        labels = inputs.get("labels")
        total_loss = task_loss
        loss_logs = {"task_loss": task_loss.item()}

        current_step = self.state.global_step
        total_steps = self.state.max_steps

        current_lambda1 = self.args.lambda1
        current_lambda2 = self.args.lambda2
        current_lambda3 = self.args.lambda3

        pt_logits = None
        if self.args.use_adaptive_regularization or self.args.lambda1 > 0 or self.args.lambda1_schedule is not None:
            if self.pt_model.device != model.device: self.pt_model.to(model.device)
            with torch.no_grad():
                pt_logits = self.pt_model(**inputs).logits
        
        if self.args.use_adaptive_regularization:
            pass
        elif self.args.lambda1_schedule == 'cosine':
            initial_l1 = self.args.lambda1
            final_l1 = 0.1 * initial_l1
            cosine_decay = 0.5 * (1 + np.cos(np.pi * current_step / total_steps))
            current_lambda1 = final_l1 + (initial_l1 - final_l1) * cosine_decay
            loss_logs["dyn_lambda1"] = current_lambda1
        
        if self.args.lambda_focus_schedule == 'two_phase':
            pass
        elif self.args.lambda_focus_schedule == 'linear_warmup':
            warmup_steps = int(self.args.lambda_warmup_ratio * total_steps)
            if current_step < warmup_steps:
                warmup_factor = current_step / warmup_steps if warmup_steps > 0 else 1.0
                current_lambda2 = self.args.lambda2 * warmup_factor
                current_lambda3 = self.args.lambda3 * warmup_factor
                loss_logs["dyn_lambda2"] = current_lambda2
                loss_logs["dyn_lambda3"] = current_lambda3
        elif self.args.lambda_focus_schedule == 'quadratic_warmup':
            warmup_steps = int(self.args.lambda_warmup_ratio * total_steps)
            if current_step < warmup_steps:
                progress = current_step / warmup_steps if warmup_steps > 0 else 1.0
                warmup_factor = progress * progress
                current_lambda2 = self.args.lambda2 * warmup_factor
                current_lambda3 = self.args.lambda3 * warmup_factor
                loss_logs["dyn_lambda2"] = current_lambda2
                loss_logs["dyn_lambda3"] = current_lambda3
        
        if current_lambda1 > 0 and pt_logits is not None:
            temp = self.args.distill_temp
            log_p_ft = F.log_softmax(ft_logits / temp, dim=-1)
            p_pt = F.softmax(pt_logits / temp, dim=-1)
            kl_mask = (labels != IGNORE_INDEX).unsqueeze(-1)
            kl_loss = (F.kl_div(log_p_ft, p_pt, reduction='none', log_target=False).sum(-1) * kl_mask.squeeze(-1)).sum() / kl_mask.sum()
            kl_loss *= (temp * temp)
            total_loss += current_lambda1 * kl_loss
            loss_logs["c_reg_loss"] = kl_loss.item()

        if current_lambda2 > 0:
            mask = (labels != IGNORE_INDEX)
            valid_logits = ft_logits[mask]
            if valid_logits.numel() > 0:
                if self.args.top_k_entropy > 0:
                    k = min(self.args.top_k_entropy, valid_logits.size(-1))
                    top_k_logits = torch.topk(valid_logits, k, dim=-1).values
                    top_k_log_probs = F.log_softmax(top_k_logits, dim=-1)
                    entropy = -(torch.exp(top_k_log_probs) * top_k_log_probs).sum(dim=-1)
                else:
                    log_probs = F.log_softmax(valid_logits, dim=-1)
                    entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
                entropy_loss = -entropy.mean()
                total_loss += current_lambda2 * entropy_loss
                loss_logs["d_reg_loss"] = entropy_loss.item()
            
        if current_lambda3 > 0:
            mask = (labels != IGNORE_INDEX).view(-1)
            valid_logits = ft_logits.view(-1, ft_logits.size(-1))[mask]
            if valid_logits.shape[0] > 1 and valid_logits.shape[1] > 1:
                try:
                    s = torch.linalg.svdvals(valid_logits)
                    k = min(self.args.svd_k, len(s))
                    sum_top_k_sv = torch.sum(s[:k])
                    norm = torch.sqrt(torch.sum(s*s)) if self.args.svd_frob_norm else torch.sum(s)
                    if norm > 1e-8:
                        svd_regularizer = - (sum_top_k_sv / norm)
                        total_loss += current_lambda3 * svd_regularizer
                        loss_logs["svd_reg_loss"] = svd_regularizer.item()
                except torch.linalg.LinAlgError as e:
                    logger.warning(f"SVD computation failed with LinAlgError. Skipping. Error: {e}")

        return (total_loss, outputs) if return_outputs else total_loss

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        return os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
    return None

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True) for text in strings]
    return {"input_ids": [np.array(t.input_ids) for t in tokenized_list]}

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, [len(s) for s in sources_tokenized["input_ids"]]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format(instruction=q) for q in examples[query]]
    targets = [f"{t}{tokenizer.eos_token}" for t in examples[response]]
    return preprocess(sources, targets, tokenizer)

def build_model(script_args, checkpoint_dir):
    compute_dtype = torch.bfloat16 if script_args.bf16 else (torch.float16 if script_args.fp16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation=script_args.attn_implementation
    )
    if not script_args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from checkpoint: {checkpoint_dir}.")
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path:
            adapter_path = os.path.join(script_args.model_name_or_path, script_args.adapter_name_or_path)
            logger.info(f"Initializing adapter from {adapter_path}.")
            try:
                with open(os.path.join(adapter_path, 'adapter_config.json'), 'r') as f:
                    adapter_config_dict = json.load(f)
            except Exception as e:
                raise IOError(f"Could not load adapter_config.json at {adapter_path}") from e
            VALID_LORA_CONFIG_KEYS = {k for k,v in LoraConfig.__dataclass_fields__.items()}
            cleaned_config_dict = {k: v for k, v in adapter_config_dict.items() if k in VALID_LORA_CONFIG_KEYS}
            peft_config = LoraConfig(**cleaned_config_dict)
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True, config=peft_config)
        else:
            logger.info('Creating new LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
                use_dora=script_args.use_dora,
            )
            model = get_peft_model(model, peft_config)
    
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()
    return model

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer_path = script_args.base_model_for_pt if script_args.base_model_for_pt else script_args.model_name_or_path
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=script_args.model_max_length,
        padding_side="right", use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)

    pt_model = None
    if script_args.use_ba_lora:
        if script_args.base_model_for_pt is None:
            raise ValueError("base_model_for_pt is required for BA-LoRA.")
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f"Loading frozen base model from: {script_args.base_model_for_pt}")
        compute_dtype = torch.bfloat16 if script_args.bf16 else torch.float16
        pt_model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.base_model_for_pt, torch_dtype=compute_dtype, trust_remote_code=True
        )

    all_training_dataset = []
    for task in script_args.sub_task:
        split_def = task.split(":")
        cur_task, cur_split = (split_def[0], f"{script_args.dataset_split}[:{split_def[1]}]") if len(split_def) > 1 else (task, script_args.dataset_split)
        ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Loaded {ds.num_rows} samples from {cur_task}/{cur_split}")
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset:
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)
    
    if torch.distributed.is_initialized() and torch.distributed.get_rank() > 0: torch.distributed.barrier()
    train_dataset = raw_train_datasets.map(
        train_tokenize_function, batched=True, batch_size=3000, num_proc=32,
        remove_columns=raw_train_datasets.column_names, desc="Tokenizing train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0: torch.distributed.barrier()
        
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    
    trainer = BALoRATrainer(
        pt_model=pt_model,
        model=model, 
        tokenizer=tokenizer, 
        args=script_args, 
        **data_module
    )
    
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(tokenizer))
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    trainer.save_state()
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir, safe_serialization=False)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        
if __name__ == "__main__":
    train()
