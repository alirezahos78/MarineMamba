import gc
import time

import torch
from fvcore.nn import FlopCountAnalysis


def compute_flops(model, input_size, device="cuda"):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    try:
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        params = sum(param.numel() for param in model.parameters())
        return total_flops, params
    except Exception as exc:
        print(f"⚠️ Error in FLOPs calculation: {exc}")
        params = sum(param.numel() for param in model.parameters())
        return 0, params
    finally:
        del dummy_input
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def compute_latency(model, input_size, device="cuda", repetitions=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.inference_mode():
        for _ in range(repetitions):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    total_time = time.time() - start_time
    del dummy_input
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return (total_time / repetitions) * 1000


def compute_memory(model, input_size, device="cuda"):
    if device != "cuda":
        return 0
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        _ = model(dummy_input)
    memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    del dummy_input
    gc.collect()
    torch.cuda.empty_cache()
    return memory_mb
