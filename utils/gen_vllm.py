import argparse
import torch
import sys
import os
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="")
parser.add_argument("--data_path", type=str, default="pissa-dataset")
parser.add_argument('--sub_task', nargs='+', help='')
parser.add_argument('--dataset_split', type=str, default="test", help='')
parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="")
parser.add_argument("--batch_size", type=int, default=400, help="")
parser.add_argument("--use_prompt_template", action=argparse.BooleanOptionalAction, default=True, help="Wrap instructions with the BA-LoRA training prompt.")
parser.add_argument('--temperature', type=float, default=0.0, help="")
parser.add_argument('--top_p', type=float, default=1, help="")
parser.add_argument('--max_tokens', type=int, default=1024, help="")
args = parser.parse_args()

stop_tokens = []
sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop=stop_tokens)
tensor_parallel_size = max(torch.cuda.device_count(), 1)
llm = LLM(model=args.model, tensor_parallel_size=tensor_parallel_size)

def batch_data(data_list, batch_size=1):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

if args.sub_task is None:
    dataset = load_dataset(args.data_path, split=args.dataset_split)
else:
    all_test_dataset = []
    for task in args.sub_task:
        ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
        print(f"{args.data_path}/{task}/{args.dataset_split}")
        for k,v in ds[0].items():
            print("-"*100)
            print(k,end=':\t')
            print(v)
        print("+"*100)
        all_test_dataset.append(ds)
        
    dataset = concatenate_datasets(all_test_dataset)
    
instructions = dataset["instruction"]
model_inputs = [PROMPT.format(instruction=item) for item in instructions] if args.use_prompt_template else instructions
batch_dataset_query = batch_data(model_inputs, batch_size=args.batch_size)
batch_dataset_answer = batch_data(dataset["output"], batch_size=args.batch_size)
batch_dataset_task = batch_data(dataset["type"], batch_size=args.batch_size)

os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
if os.path.exists(args.output_file):
    os.remove(args.output_file)

for idx, (batch_query, batch_answer, batch_task) in enumerate(zip(batch_dataset_query, batch_dataset_answer,batch_dataset_task)):
    with torch.no_grad():
        completions = llm.generate(batch_query, sampling_params)
    for query, completion, answer, task in zip(batch_query, completions, batch_answer, batch_task):
        with open(args.output_file, 'a', encoding="utf-8") as f:
            json.dump({'type': task, 'query': query, 'output': completion.outputs[0].text, 'answer': answer}, f)
            f.write('\n')
