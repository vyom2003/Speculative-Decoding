import torch
import random
from transformers import AutoTokenizer, T5ForConditionalGeneration
import time
import numpy as np

from utils import calculate_optimal_gamma, gamma_tokens, generate_token, sample

def speculative_beam_decode(input_text, target_model_name="google-t5/t5-large", small_model_name="google-t5/t5-small", gamma=3, 
                            temp=0, dynamicGamma = False, cost = 0.1, beam_width=3, max_length=50):
    model_small = T5ForConditionalGeneration.from_pretrained(small_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = T5ForConditionalGeneration.from_pretrained(target_model_name)

    input_ids = target_tokenizer(input_text, return_tensors="pt", truncation=False).input_ids
    initial_prefix = torch.tensor([[model_small.config.pad_token_id]])
    beams = [(initial_prefix, 0.0)]  # Each beam is a tuple (prefix, score)
    completed_hypotheses = []
    total_time = 0
    num_ops = 0
    gamma_Arr = []
    beta_rate = []

    if dynamicGamma:
        gamma = calculate_optimal_gamma(0.7, cost)
    
    while len(beams) > 0:
        new_beams = []
        for prefix, score in beams:
            if prefix[0][-1].item() == target_model.config.eos_token_id or len(prefix[0]) >= max_length:
                completed_hypotheses.append((prefix, score))
                continue

            generated_logits, generated_tokens = gamma_tokens(model_small, prefix, input_ids, temp, gamma)
            target_prefixes = torch.cat([prefix, torch.tensor([generated_tokens])], dim=1)

            logits_mp = generate_token(target_model, input_ids=input_ids, prefix=target_prefixes, temp=temp)
            total_time += time.time() - start_time

            start_time = time.time()
            n, new_tokens = 0, []

            for i in range(gamma):
                r = random.uniform(0, 1)
                q = generated_logits[i][0][-1][generated_tokens[i].item()].item()
                p = logits_mp[0][i][generated_tokens[i].item()].item()
                if r > p / q:
                    break
                n += 1
                new_tokens.append(generated_tokens[i].item())

            beta_rate.append(n / gamma)
            p_prime = logits_mp[0][-1]

            if n < gamma:
                p_prime = torch.clamp(logits_mp[0][n] - generated_logits[n][0][-1], min=0)
                p_prime = p_prime / p_prime.sum(dim=-1, keepdim=True)

            p_prime = p_prime.unsqueeze(0).unsqueeze(0)
            last_token_id = sample(p_prime, temperature=temp).item()
            new_tokens.append(last_token_id)

            num_ops += gamma * cost + 1

            new_prefix = torch.cat([prefix, torch.tensor([new_tokens])], dim=1)
            new_beams.append((new_prefix, score - np.log(p_prime[0, 0, last_token_id].item())))

            total_time += time.time() - start_time

        beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]

        if dynamicGamma:
            gamma_Arr.append(gamma)
            gamma = calculate_optimal_gamma(np.mean(np.array(beta_rate)), cost)

    completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x[1])
    best_hypothesis = completed_hypotheses[0] if completed_hypotheses else beams[0]

    del model_small
    del target_model

    return target_tokenizer.decode(best_hypothesis[0][0].tolist(), skip_special_tokens=True), total_time, np.mean(np.array(beta_rate)),  num_ops

