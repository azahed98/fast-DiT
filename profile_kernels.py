import os
import torch

from accelerate import Accelerator
from glob import glob

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from train import create_logger

from torch.profiler import profile, ProfilerActivity
import argparse

from dataclasses import dataclass

@dataclass
class KernelArgs:
    warmup_steps: int = 1
    model: torch.nn.Module = None
    diffusion: torch.nn.Module = None
    accelerator: Accelerator = None
    bs: int = None
    latent_size: int = None
    backwards: bool = True

def profile_training_loss_learned_var(kernelArgs: KernelArgs):
    diffusion = kernelArgs.diffusion
    accelerator = kernelArgs.accelerator
    bs = kernelArgs.bs
    latent_size = kernelArgs.latent_size

    device = accelerator.device

    C = 4
    x_start = torch.randn((bs, C, latent_size, latent_size), device=device)
    x_t = torch.randn((bs, C, latent_size, latent_size), device=device)
    t = torch.randint(0, diffusion.num_timesteps, (bs,), device=device)
    terms = {}
    model_output = torch.randn((bs, C * 2, *x_t.shape[2:]), device=device)
    
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA
    #     ],
    #     schedule=torch.profiler.schedule(wait=kernelArgs.warmup_steps, warmup=0, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{experiment_dir}/_training_loss_learned_var_forward"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    # ) as prof:
    #     for _ in range(kernelArgs.warmup_steps):
    #         result = diffusion._training_loss_learned_var(x_start, x_t, t, terms, model_output)
    #         prof.step()
    #     result = diffusion._training_loss_learned_var(x_start, x_t, t, terms, model_output)
    #     prof.step()
    result = diffusion._training_loss_learned_var(x_start, x_t, t, terms, model_output)
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        #schedule=torch.profiler.schedule(wait=kernelArgs.warmup_steps, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{experiment_dir}/_training_loss_learned_var_forward"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        # for _ in range(kernelArgs.warmup_steps):
        #     result = diffusion._training_loss_learned_var(x_start, x_t, t, terms, model_output)
        #     prof.step()
        result = diffusion._training_loss_learned_var(x_start, x_t, t, terms, model_output)
        prof.step()

    # if kernelArgs.backwards:
    #     result[0]["vb"].backward()

KERNEL_TEST_FUNCTIONS = {
    "training_loss_learned_var" : profile_training_loss_learned_var,
}

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", type=str, nargs='+', help="List of kernels to profile", default=["all"])
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=1)

    args = parser.parse_args()
    
    testing_kernels = args.kernels
    if testing_kernels == ["all"]:
        testing_kernels = list(KERNEL_TEST_FUNCTIONS.keys())
    
    for kernel in testing_kernels:
        assert kernel in KERNEL_TEST_FUNCTIONS, f"Invalid kernel {kernel}"
    
    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    diffusion = create_diffusion(timestep_respacing="") 
    
    model = accelerator.prepare(model)

    kernelArgs = KernelArgs(
        warmup_steps=1,
        model=model,
        diffusion=diffusion,
        accelerator=accelerator,
        bs=args.global_batch_size,
        latent_size=latent_size
    )

    # Init profiler
    for kernel in testing_kernels:
        KERNEL_TEST_FUNCTIONS[kernel](kernelArgs)