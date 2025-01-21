from speculative_decoding import speculative_decode
from utils import calculate_bleu
from speculative_decoding_heirarchy import  speculative_decode_hierarchy
from speculative_decoding_beam_search import speculative_beam_decode
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import warnings
import numpy as np
from baseline_utils import baseline_decode
import argparse
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parsing_arguments():

    parser = argparse.ArgumentParser(description="Parsing Trading params")
        
    parser.add_argument('--target_model', type=str, help='Target Model')
    parser.add_argument('--small_model', type=str, help='Small Model')
    parser.add_argument('--dynamic_gamma', type=str, help='Dynamic Gamma')
    parser.add_argument('--heirarchy', type=str, help="Enable heirarchy")
    parser.add_argument('--beam_decode', type=str, help='Beam Decode')

    args = parser.parse_args()

    return args.target_model, args.small_model, args.dynamic_gamma, args.heirarchy, args.beam_decode

if __name__ == "__main__":
    target_model, small_model, dynamic_gamma, heirarchical, beam_decode = parsing_arguments()

    ds = load_dataset("bentrevett/multi30k")
    ds = ds["test"].select(range(1))
    test_dl = DataLoader(ds)

    if target_model == 'large':
        target_model = "google-t5/t5-large"
    if small_model == 'small':
        small_model = "google-t5/t5-small"
    elif small_model == 'base':
        small_model = "google-t5/t5-base"
    
    if dynamic_gamma == 'true':
        dynamic_gamma = True
    else:
        dynamic_gamma = False
    
    if heirarchical == 'true':
        heirarchical = True
    else:
        heirarchical = False
    
    if beam_decode == 'true':
        beam_decode = True
    else:
        beam_decode = False
    
    target_model_bleu_arr=[]
    target_model_inference_time_arr=[]
    target_model_num_ops_arr=[]

    for input_sentence in tqdm(test_dl):
        input_prompt = f"Translate to German: {input_sentence['en']}"
        print(f"INPUT PROMPT: {input_prompt}")
        output_text, total_time, ops = baseline_decode(input_text=input_prompt, model_name=target_model)
        target_model_bleu_arr.append(calculate_bleu(input_sentence["de"][0],output_text))
        target_model_inference_time_arr.append(total_time)
        target_model_num_ops_arr.append(ops)
    
    target_model_bleu = np.mean(np.array(target_model_bleu_arr))
    target_model_inference = np.mean(np.array(target_model_inference_time_arr))
    target_model_num_ops = np.mean(np.array(target_model_num_ops_arr))

    small_model_bleu_arr=[]
    small_model_inference_time_arr=[]
    small_model_num_ops_arr=[]

    for input_sentence in tqdm(test_dl):
        input_prompt = f"Translate to German: {input_sentence['en']}"
        output_text, total_time, ops = baseline_decode(input_text=input_prompt, model_name=small_model)
        small_model_bleu_arr.append(calculate_bleu(input_sentence["de"][0],output_text))
        small_model_inference_time_arr.append(total_time)
        small_model_num_ops_arr.append(ops)
    
    small_model_bleu = np.mean(np.array(small_model_bleu_arr))
    small_model_inference = np.mean(np.array(small_model_inference_time_arr))
    small_model_num_ops = np.mean(np.array(small_model_num_ops_arr))

    print(f"Average Inference Time for Target Model : {target_model_inference}")
    print(f"Average Inference Time for Small Model : {small_model_inference}")

    print(f"Average Operation Taken for Target Model : {target_model_num_ops}")
    print(f"Average Operation Taken for Small Model : {small_model_num_ops}")

    print(f"Average Bleu Score for Target Model : {target_model_bleu}")
    print(f"Average Bleu Score for Small Model : {small_model_bleu}")

    c = small_model_inference/target_model_inference
    print(f"Ratio of Inference Time for Target and Small Model : {c}")
    
    inference_time = []
    all_alphas=[]
    bleu_arr=[]
    num_entries_arr=[]
    for input_sentence in tqdm(test_dl):
        input_prompt = f"Translate to German: {input_sentence['en']}"
        if beam_decode:
            beam_width = 3
            max_length = 50
            output_text, total_time, alpha, num_entries = speculative_beam_decode(input_text=input_prompt, target_model_name=small_model, dynamicGamma=dynamic_gamma, cost=c, beam_width=beam_width, max_length=max_length)
        elif heirarchical:
            output_text, total_time, alpha, num_entries = speculative_decode_hierarchy(input_prompt)
        else:
            output_text, total_time, alpha, num_entries = speculative_decode(input_text=input_prompt, target_model_name=target_model, small_model_name=small_model, dynamicGamma=dynamic_gamma, cost=c)
        bleu_arr.append(calculate_bleu(input_sentence["de"][0],output_text))
        inference_time.append(total_time)
        all_alphas.append(alpha)
        num_entries_arr.append(num_entries)

    print("Average BLEU Score: ", np.mean(np.array(bleu_arr)))
    print("Alpha: ", np.mean(np.array(all_alphas)))
    
    average_entries = np.mean(np.array(num_entries_arr))
    # print(f"Speed Up Ratio of Method : {target_model_num_ops/average_entries}")
    average_inference_time = np.mean(np.array(inference_time))
    print(f"Average Inference Time: {average_inference_time}")
    print(f"Speed Up Ratio of Method : {target_model_inference/average_inference_time}")