import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from key import key
import tqdm

from datasets import load_dataset
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="../act_scales/llama-2-7b.pt",
)
parser.add_argument("--dataset_name", type=str, default="wikitext2")
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--quantize_bonus", action="store_true")
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = args.act_scales_path
n_samples = args.n_samples


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40, eval_dataset_name='wikitext2'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        if eval_dataset_name == 'wikitext2':
            self.dataset = tokenizer(
                "\n\n".join(dataset["text"]), return_tensors="pt"
            ).input_ids.to(device)
        elif eval_dataset_name == 'ptb':
            self.dataset = tokenizer(
                " ".join(dataset["sentence"]), return_tensors="pt"
            ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        print(n_samples)
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


tokenizer = AutoTokenizer.from_pretrained(model_path, token=key)
eval_dataset_name = args.dataset_name
if eval_dataset_name == 'wikitext2':
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", verification_mode='no_checks')
elif eval_dataset_name == 'ptb':
    dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples, eval_dataset_name=eval_dataset_name)

#model = AutoModelForCausalLM.from_pretrained(
    #model_path, torch_dtype=torch.bfloat16, device_map="auto"
#)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", token=key
)

if args.smooth:
    print("using smooth")
    st = time.time()
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)
    print(f"smooth took {time.time()-st}s")

if args.quantize:
    print("quantizing")
    st = time.time()
    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
    )
    print(f"quantization took {time.time()-st}s")

ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")

# eval on ptb also
dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples, eval_dataset_name='ptb')
ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")


if args.save_model:
    model_path = "../../models/" + args.save_model
    os.makedirs(model_path, exist_ok=True)
    print(f"saving model to path: {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

