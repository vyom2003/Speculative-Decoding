import torch
import random
from transformers import AutoTokenizer, T5ForConditionalGeneration
import time
import numpy as np

from utils import gamma_tokens, generate_token, sample

def speculative_decode_hierarchy(
    input_text, 
    target_model_name="google-t5/t5-large", 
    small_model_name="google-t5/t5-base", 
    medium_model_name="google-t5/t5-small", 
    gamma=5, 
    temp=0, 
    cost=0.1
):
    model_small = T5ForConditionalGeneration.from_pretrained(small_model_name)
    model_medium = T5ForConditionalGeneration.from_pretrained(medium_model_name)
    target_model = T5ForConditionalGeneration.from_pretrained(target_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    input_ids = target_tokenizer(input_text, return_tensors="pt", truncation=False).input_ids
    prefix = torch.tensor([[target_model.config.pad_token_id]])
    
    total_time = 0
    beta_rate = []
    num_entries = 0

    while prefix[0][-1].item() != target_model.config.eos_token_id:
        num_entries += gamma * cost + 1
        start_time = time.time()
        small_generated_logits, small_generated_tokens = gamma_tokens(model_small, prefix, input_ids, temp, gamma)
        total_time += time.time() - start_time
        medium_prefixes = torch.cat([prefix, torch.tensor([small_generated_tokens])], dim=1)
        start_time = time.time()
        logits_mp = generate_token(model_medium, input_ids=input_ids, prefix=medium_prefixes, temp=temp)
        total_time += time.time() - start_time
        n, new_tokens = 0, []
        generated_logits = torch.tensor([])
        generated_tokens = torch.tensor([])
        for i in range(gamma):
            r = random.uniform(0, 1)
            q = small_generated_logits[i][0][-1][small_generated_tokens[i].item()].item()
            p = logits_mp[0][prefix.shape[-1]-1+i][small_generated_tokens[i].item()].item()
            if r > p / q:
                break
            n += 1
            new_tokens.append(small_generated_tokens[i].item())
            generated_logits = torch.cat([
                generated_logits, 
                torch.tensor(small_generated_logits[i][0][-1]).unsqueeze(0).unsqueeze(0)
            ])
            generated_tokens = torch.cat([generated_tokens,small_generated_tokens[i].unsqueeze(0)])
        start_time = time.time()
        medium_generated_logits, medium_generated_tokens = gamma_tokens(model_medium, torch.cat([prefix, torch.tensor([new_tokens])], dim=1).to(torch.long), input_ids, temp, gamma-n)
        total_time += time.time() - start_time
        if(n<gamma):
            generated_logits = torch.cat([
                generated_logits, 
                torch.stack([torch.tensor(logit[0][-1]).unsqueeze(0) for logit in medium_generated_logits])
            ], dim=0)
            generated_tokens = torch.cat([
                generated_tokens, 
                torch.stack([torch.tensor(token).unsqueeze(0) for token in medium_generated_tokens],dim=1).squeeze(0)
            ], dim=0)

        target_prefixes = torch.cat([torch.cat([prefix, torch.tensor([new_tokens])], dim=1), torch.tensor([medium_generated_tokens])], dim=1)
        start_time = time.time()
        logits_mp = generate_token(target_model, input_ids=input_ids, prefix=target_prefixes.to(torch.long), temp=temp)
        total_time += time.time() - start_time
        n, new_tokens = 0, []

        generated_tokens = generated_tokens.to(torch.int)
        for i in range(gamma):
            r = random.uniform(0, 1)
            q = generated_logits[i][0][generated_tokens[i].item()].item()
            p = logits_mp[0][prefix.shape[-1]-1+i][generated_tokens[i].item()].item()
            if r > p / q:
                break
            n += 1
            new_tokens.append(generated_tokens[i].item())

        beta_rate.append(n/gamma)
        p_prime = logits_mp[0][-1]
        if n < gamma:
            p_prime = torch.clamp(logits_mp[0][prefix.shape[-1]-1+n] - generated_logits[n][0], min=0)
            p_prime = p_prime / p_prime.sum(dim=-1, keepdim=True)

        p_prime = p_prime.unsqueeze(0).unsqueeze(0)
        last_token_id = sample(p_prime, temperature=temp).item()
        new_tokens.append(last_token_id)
        prefix = torch.cat([prefix, torch.tensor([new_tokens])], dim=1)
        print(target_tokenizer.decode(prefix[0].tolist(), skip_special_tokens=True))
    del model_small
    del model_medium
    del target_model
    
    return (
        target_tokenizer.decode(prefix[0].tolist(), skip_special_tokens=True), 
        total_time, 
        np.mean(np.array(beta_rate)) if beta_rate else 0, 
        num_entries
    )