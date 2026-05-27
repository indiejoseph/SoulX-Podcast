import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os
import time
from pathlib import Path
import torch
from transformers import RepetitionPenaltyLogitsProcessor

from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import process_single_input, initiate_model

def profile():
    seed = 198964
    model_path = "pretrained_models/SoulX-Podcast-1.7B-dialect"
    
    # Verify model directory exists
    if not os.path.exists(model_path):
        print(f"[Error] Model path {model_path} not found.")
        return

    print("[Profiler] Loading model (hf engine)...")
    load_start = time.time()
    model, dataset = initiate_model(seed, model_path, llm_engine="hf", fp16_flow=True)
    print(f"[Profiler] Model loaded in {time.time() - load_start:.2f}s")
    
    S1_PROMPT_WAV = Path("example/audios/female_mandarin.wav")
    spk1_prompt_text = "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
    
    data = {
        "speakers": {
            "S1":{
                "prompt_audio": S1_PROMPT_WAV,
                "prompt_text": spk1_prompt_text,
            }
        },
        "text": [
            ["S1", "Hello everyone, welcome to our show. Hey, I want to ask you something today, do you feel that, um, driving an electric car nowadays, the most annoying, most annoying thing is what?"]
        ]
    }
    
    print("[Profiler] Preparing inputs (Frontend)...")
    frontend_t0 = time.time()
    inputs = podcast_format_parser(data)
    processed_data = process_single_input(
        dataset,
        inputs['text'],
        inputs['prompt_wav'],
        inputs['prompt_text'],
        inputs['use_dialect_prompt'],
        inputs['dialect_prompt_text'],
    )
    frontend_time = time.time() - frontend_t0
    
    # Profiling variables
    llm_start_time = None
    first_token_time = None
    llm_times = []
    flow_times = []
    hift_times = []
    
    # 1. Patch LogitsProcessor for TTFT
    # RepetitionPenaltyLogitsProcessor is the first one applied in HFLLMEngine.
    orig_rp_call = RepetitionPenaltyLogitsProcessor.__call__
    def patched_rp_call(self, input_ids, scores):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()
        return orig_rp_call(self, input_ids, scores)
    RepetitionPenaltyLogitsProcessor.__call__ = patched_rp_call
    
    # 2. Patch LLM Generate
    original_llm_generate = model.llm.generate
    def wrapped_llm_generate(*args, **kwargs):
        nonlocal llm_start_time, first_token_time
        llm_start_time = time.time()
        first_token_time = None # reset for each turn
        res = original_llm_generate(*args, **kwargs)
        llm_times.append({
            "total": time.time() - llm_start_time,
            "ttft": first_token_time - llm_start_time if first_token_time else 0,
            "num_new_tokens": len(res['token_ids'])
        })
        return res
    model.llm.generate = wrapped_llm_generate
    
    # 3. Patch Flow
    original_flow_forward = model.flow.forward
    def wrapped_flow_forward(*args, **kwargs):
        t0 = time.time()
        res = original_flow_forward(*args, **kwargs)
        flow_times.append(time.time() - t0)
        return res
    model.flow.forward = wrapped_flow_forward
    
    # 4. Patch HiFT (Vocoder)
    original_hift_forward = model.hift.forward
    def wrapped_hift_forward(*args, **kwargs):
        t0 = time.time()
        res = original_hift_forward(*args, **kwargs)
        hift_times.append(time.time() - t0)
        return res
    model.hift.forward = wrapped_hift_forward
    
    print("[Profiler] Running inference (profiling)...")
    total_t0 = time.time()
    results_dict = model.forward_longform(**processed_data)
    total_time = time.time() - total_t0
    
    wavs = results_dict['generated_wavs']
    total_audio_length_sec = sum([w.shape[-1] for w in wavs]) / 24000.0
    
    print("\n" + "="*50)
    print("PROFILING RESULTS")
    print("="*50)
    print(f"Total Audio Generated  : {total_audio_length_sec:.3f} s")
    print(f"Frontend Preprocessing : {frontend_time:.3f} s  <-- (Text norm, Spk Embed, Mel)")
    print(f"Total Inference Time   : {total_time:.3f} s")
    print(f"Overall RTF            : {total_time / total_audio_length_sec:.3f}")
    
    if len(llm_times) > 0:
        print("\n--- LLM Stage ---")
        tot_llm = sum(t["total"] for t in llm_times)
        tot_ttft = sum(t["ttft"] for t in llm_times)
        tot_tokens = sum(t["num_new_tokens"] for t in llm_times)
        print(f"Total LLM Time         : {tot_llm:.3f} s")
        print(f"Avg TTFT               : {tot_ttft / len(llm_times):.3f} s")
        decode_time = tot_llm - tot_ttft
        print(f"Avg Decode Time        : {decode_time / len(llm_times):.3f} s")
        print(f"Tokens Generated       : {tot_tokens}")
        print(f"Decode Speed (tok/s)   : {tot_tokens / decode_time if decode_time > 0 else 0:.2f} tok/s")
        print(f"LLM RTF                : {tot_llm / total_audio_length_sec:.3f}")
    
    if len(flow_times) > 0:
        print("\n--- Flow (Diffusion) Stage ---")
        print(f"Total Flow Time        : {sum(flow_times):.3f} s")
        print(f"Flow RTF               : {sum(flow_times) / total_audio_length_sec:.3f}")
        
    if len(hift_times) > 0:
        print("\n--- Vocoder (HiFT) Stage ---")
        print(f"Total HiFT Time        : {sum(hift_times):.3f} s")
        print(f"HiFT RTF               : {sum(hift_times) / total_audio_length_sec:.3f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    profile()
