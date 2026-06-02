import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import set_seed
from diffusers import StableDiffusionXLPipeline
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from ultralytics import YOLO

from composition import (
    get_inpaint_model,
    multiobj_conditional_composition,
)
from decomposition import (
    attnScore_decomposition,
    get_fastrcnn,
    objDetect_decomposition,
)
from sdxl_pipelines import (
    AttendAndExciteSDXL,
    ComposableSDXL,
    LayoutGuidanceSDXL,
    RegisterAttnSDXL,
    StructureGuidanceSDXL,
)
from utils import (
    get_tok2id_map,
    get_token_ids_and_bboxes,
    get_token_indices,
    load_coco_filtered,
    prepare_condComp_prompts,
)


def is_dist() -> bool:
    """Return True if torch.distributed is available and initialized.

    Returns:
        bool: Whether distributed mode is available and initialized.
    """

    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the world size (number of processes) for distributed runs.

    Returns:
        int: Number of processes in the distributed group, or 1 if not distributed.
    """

    return dist.get_world_size() if is_dist() else 1


def get_kwargs_for_pipe_by_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct keyword arguments for SDXL pipelines from parsed CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Mapping of keyword arguments to pass to pipeline constructors.
    """

    kwargs_for_pipe: Dict[str, Any] = {}

    if getattr(args, "scale_factor", None) is not None:
        kwargs_for_pipe["scale_factor"] = args.scale_factor

    if getattr(args, "opt_layers", None) is not None:
        kwargs_for_pipe["optimized_layers"] = list(
            range(args.opt_layers[0], args.opt_layers[1])
        )

    return kwargs_for_pipe


def main(
    args: argparse.Namespace,
    total_images: int,
    out_dir: str,
    seed: int = 0,
    batch: int = 3,
) -> None:
    """Main entrypoint to generate images using SDXL pipelines in a distributed setup.

    Args:
        args (argparse.Namespace): Parsed CLI arguments controlling pipelines and options.
        total_images (int): Total number of images to generate across all processes.
        out_dir (str): Output directory where generated images and captions are saved.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        batch (int, optional): Number of images to generate per inner loop iteration. Defaults to 3.

    Returns:
        None
    """
    state = PartialState()  # handles world_size/rank/device
    os.makedirs(out_dir, exist_ok=True)

    # Split workload
    per_rank = math.ceil(total_images / state.num_processes)
    start = state.process_index * per_rank
    end = min(total_images, start + per_rank)

    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if start >= end:
        return  # this rank has no work

    # Print args only on rank 0
    if state.process_index == 0:
        print(args)

    # One pipeline per process, on its own GPU
    if args.compose_sdxl:
        print("Using Composable SDXL pipeline.")
        pipe = ComposableSDXL.from_single_file(
            args.sdxl_ckpt,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(state.device)
    elif args.attend_and_excite_sdxl:
        print("Using Attend and Excite SDXL pipeline.")
        pipe = AttendAndExciteSDXL.from_single_file(
            args.sdxl_ckpt,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(state.device)
        pipe.efficient_mode()
    elif args.layout_guidance_sdxl:
        print("Using Layout Guidance SDXL pipeline.")
        pipe = LayoutGuidanceSDXL.from_single_file(
            args.sdxl_ckpt,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(state.device)
        pipe.efficient_mode()
    elif args.structure_guidance_sdxl:
        print("Using Structure Guidance SDXL pipeline.")
        pipe = StructureGuidanceSDXL.from_single_file(
            args.sdxl_ckpt,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(state.device)
        pipe.efficient_mode()
    elif args.condComp:
        print("Using Conditional Composition SDXL pipeline.")
        if args.ae_for_joint_stage:
            print(
                "Using Attend and Excite SDXL for joint stage in Conditional Composition."
            )
            sdxl_pipe = AttendAndExciteSDXL.from_single_file(
                args.sdxl_ckpt,
                torch_dtype=torch.float16,
                add_watermarker=False,
            ).to(state.device)
            sdxl_pipe.efficient_mode()
        else:
            sdxl_pipe = RegisterAttnSDXL.from_single_file(
                args.sdxl_ckpt,
                torch_dtype=torch.float16,
                add_watermarker=False,
            ).to(state.device)

        inpaint_pipe = get_inpaint_model(state.device, sdxl_pipe)
        sdxl_pipe.set_progress_bar_config(disable=True)
        inpaint_pipe.set_progress_bar_config(disable=True)

        # set different generators to avoid consumption of the main generator
        inpaint_gen = torch.Generator(device=state.device).manual_seed(
            seed + state.process_index + 1126
        )
        # resample_gen = torch.Generator(device=state.device).manual_seed(
        #     seed + state.process_index + 6211
        # )

        if args.attnScore_decomp:
            print("Using decomposition estimates by cross attention scores")
        if args.objDetect_decomp:
            print(
                "Using decomposition estimates by object detector (Faster R-CNN) scores."
            )
            fastrcnn = get_fastrcnn(device=state.device)
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            args.sdxl_ckpt,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(state.device)

    if not args.condComp:
        pipe.set_progress_bar_config(disable=True)

    # Get kwargs for pipeline
    kwargs_for_pipe = get_kwargs_for_pipe_by_args(args)

    captions = None
    if state.process_index == 0:
        if args.compose_sdxl and args.use_coco_caption:
            raise ValueError("Cannot use COCO captions with composable SDXL.")
        captions = load_coco_filtered(
            instances_json=args.coco_instances_json,
            class_count=args.class_count,
            total_images=total_images,
            one_caption_per_image="first",
            shuffle=True,
            use_coco_caption=args.use_coco_caption,
            compose_class=args.compose_sdxl,
            seed=seed,
        )

    # Broadcast to all ranks
    obj = [captions]
    if get_world_size() > 1:  # we only broadcast objects if gpu number > 1
        dist.broadcast_object_list(obj, src=0)

    captions = obj[0]

    gen = torch.Generator(device=state.device).manual_seed(seed + state.process_index)

    # Only show tqdm for rank 0 (avoid multiple bars)
    progress = tqdm(
        total=end - start,
        desc=f"Rank {state.process_index} generating images",
        disable=(state.process_index != 0),
    )

    def sample_latents(
        size: Tuple[int, ...], generator: torch.Generator
    ) -> torch.Tensor:
        """Generate a tensor of random latents for the model.

        Args:
            size (Tuple[int, ...]): Shape of the latent tensor to generate (e.g., (1,4,128,128)).
            generator (torch.Generator): Generator used for reproducible random draws.

        Returns:
            torch.Tensor: Random tensor on the target device with dtype float16.
        """

        return randn_tensor(
            size,
            generator=generator,
            device=state.device,
            dtype=torch.float16,
        )

    i = start
    local_captions = []
    while i < end:
        torch.cuda.empty_cache()
        n = min(batch, end - i)  # <= MAX_IMAGES_PER_PROMPT
        caption_list_batch = [c["caption"] for c in captions[i : i + n]]
        id_batch = [c["image_id"] for c in captions[i : i + n]]
        categories_batch = [c["categories"] for c in captions[i : i + n]]

        images = []
        if args.compose_sdxl:
            for cap in caption_list_batch:
                latents = sample_latents((1, 4, 128, 128), generator=gen)
                assert (
                    "|" in cap
                ), "Composable SDXL requires '|' in prompt for conditioning."
                images.append(
                    pipe(prompt=cap, num_images_per_prompt=1, latents=latents).images[0]
                )
        elif args.structure_guidance_sdxl:
            for cap in caption_list_batch:
                latents = sample_latents((1, 4, 128, 128), generator=gen)
                images.append(
                    pipe(prompt=cap, num_images_per_prompt=1, latents=latents).images[0]
                )

        elif args.layout_guidance_sdxl:
            for j, (cap, cat) in enumerate(zip(caption_list_batch, categories_batch)):
                idx_map = get_tok2id_map(pipe, cap)
                latents = sample_latents((1, 4, 128, 128), generator=gen)
                token_indices, bboxes, target_tokens = get_token_ids_and_bboxes(
                    idx_map,
                    cat,
                    img_size=pipe.default_sample_size * pipe.vae_scale_factor,
                    tokenizer=pipe.tokenizer,
                    return_tokens=True,
                    ignore_missing=False,
                )
                captions[i + j]["bboxes"] = bboxes

                if len(token_indices) <= 0:
                    print(
                        f"Warning: No token indices found for categories {cat} in prompt: {cap}. Skip this image."
                    )
                    continue
                if len(token_indices) != len(bboxes):
                    print(
                        f"Warning: Mismatched token indices and bboxes, # of tokens: {len(token_indices)} vs # of bboxes: {len(bboxes)}. Skip this image."
                    )
                    continue

                _img = pipe(
                    prompt=cap,
                    token_indices=token_indices,
                    bboxes=bboxes,
                    latents=latents,
                    generator=gen,
                    num_images_per_prompt=1,
                    num_inference_steps=50,
                    attn_res=(32, 32),
                    **kwargs_for_pipe,
                ).images[0]
                _img_path = os.path.join(out_dir, f"sdxl_id{id_batch[j]}.png")
                _img.save(_img_path)
                images.append(_img)
                torch.cuda.empty_cache()

        elif args.attend_and_excite_sdxl:
            for j, (cap, cat) in enumerate(zip(caption_list_batch, categories_batch)):
                latents = sample_latents((1, 4, 128, 128), generator=gen)
                idx_map = get_tok2id_map(pipe, cap)
                token_indices, target_tokens = get_token_indices(
                    idx_map,
                    cat,
                    tokenizer=pipe.tokenizer,
                    return_tokens=True,
                )
                if len(token_indices) <= 0:
                    print(
                        f"Warning: No token indices found for categories {cat} in prompt: {cap}. Skip this image."
                    )
                    continue
                _img = pipe(
                    prompt=cap,
                    token_indices=token_indices,
                    num_images_per_prompt=1,
                    latents=latents,
                    generator=gen,
                    attn_res=(32, 32),
                    add_joint_composition_loss=args.add_joint_composition_loss,
                    categories=cat,
                    tok_map=idx_map,
                    **kwargs_for_pipe,
                ).images[0]
                _img_path = os.path.join(out_dir, f"sdxl_id{id_batch[j]}.png")
                _img.save(_img_path)
                images.append(_img)
                torch.cuda.empty_cache()

        elif args.condComp:
            decomposition_size = args.decomposition_size
            for j, (cap, cat) in enumerate(zip(caption_list_batch, categories_batch)):
                latents = sample_latents((1, 4, 128, 128), generator=gen)
                inpaint_noise = sample_latents((1, 4, 128, 128), generator=inpaint_gen)
                if args.drop_stage_thres < 1.0:
                    drop_probs = [random.random()]
                    drop_stages = [False] + [
                        prob > args.drop_stage_thres for prob in drop_probs
                    ]
                    captions[i + j]["drop_probs"] = drop_probs
                else:
                    drop_stages = None

                if args.objDetect_decomp:
                    _img_per_stage, _cat, scores = objDetect_decomposition(
                        sdxl_pipe=sdxl_pipe,
                        inpaint_pipe=inpaint_pipe,
                        categories=cat,
                        keep_ratio_offset=args.keep_ratio_offset,
                        latents=latents,
                        inpaint_noise=inpaint_noise,
                        object_detector=fastrcnn,
                        strength=args.strength,
                        drop_stages=drop_stages,
                        use_ae_for_joint_stage=args.ae_for_joint_stage,
                    )
                    _img = _img_per_stage[-1]
                    captions[i + j]["scores"] = scores
                    prompts_list, categories_list, keep_ratios = (
                        prepare_condComp_prompts(
                            cat,
                            _cat,
                            keep_ratio_offset=args.keep_ratio_offset,
                        )
                    )
                elif args.attnScore_decomp:
                    _img_per_stage, _cat, scores = attnScore_decomposition(
                        sdxl_pipe,
                        inpaint_pipe,
                        cat,
                        args.keep_ratio_offset,
                        latents,
                        inpaint_noise,
                        t_joint=args.t_joint,
                        t_comp=args.t_comp,
                        resample_steps=args.resample_steps,
                        layout_guidance=not args.wo_layout_guidance,
                        strength=args.strength,
                        return_scores=True,
                        attn_threshold=args.attn_threshold,
                        decomposition_size=decomposition_size,
                        drop_stages=drop_stages,
                    )
                    _img = _img_per_stage[-1]
                    prompts_list, categories_list, keep_ratios = (
                        prepare_condComp_prompts(
                            cat,
                            _cat,
                            keep_ratio_offset=args.keep_ratio_offset,
                        )
                    )
                else:
                    # randomly pick categories to add if not using success rate sorting
                    _cat = random.sample(cat, k=min(decomposition_size, len(cat)))
                    prompts_list, categories_list, keep_ratios = (
                        prepare_condComp_prompts(
                            cat,
                            _cat,
                            keep_ratio_offset=args.keep_ratio_offset,
                        )
                    )
                    _img = multiobj_conditional_composition(
                        sd_pipe=sdxl_pipe,
                        inpaint_pipe=inpaint_pipe,
                        prompts_list=prompts_list,
                        categories_list=categories_list,
                        generator=gen,
                        latents=latents,
                        inpaint_noise=inpaint_noise,
                        keep_ratios=keep_ratios,
                        resample_steps=args.resample_steps,
                        strength=args.strength,
                        use_layout_guidance=not args.wo_layout_guidance,
                    )

                captions[i + j]["categories_list"] = categories_list
                captions[i + j]["prompts_list"] = prompts_list
                captions[i + j]["keep_ratios"] = keep_ratios
                images.append(_img)
        else:  # SDXL
            for cap in caption_list_batch:
                latents = sample_latents((1, 4, 128, 128), generator=gen)
                images.append(
                    pipe(
                        prompt=cap,
                        num_images_per_prompt=1,
                        latents=latents,
                    ).images[0]
                )

        # Store captions in each rank
        if captions is not None:
            local_captions.extend(captions[i : i + n])

        # Save images
        for img_id, img in zip(id_batch, images):
            if img is not None:
                img.save(os.path.join(out_dir, f"sdxl_id{img_id}.png"))
            i += 1

        progress.update(len(images))

    if state.num_processes > 1:
        gathered = (
            [None for _ in range(state.num_processes)]
            if state.process_index == 0
            else None
        )

        dist.gather_object(local_captions, gathered, dst=0)

        if state.process_index == 0:
            merged_captions = []
            for part in gathered:
                if part is not None:
                    merged_captions.extend(part)

            json_path = os.path.join(out_dir, "coco_captions.json")
            with open(json_path, "w") as f:
                json.dump(merged_captions, f, indent=4)

            print(f"Saved COCO captions to: {json_path}")

    else:
        json_path = os.path.join(out_dir, "coco_captions.json")
        with open(json_path, "w") as f:
            json.dump(local_captions, f, indent=4)

        print(f"Saved COCO captions to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed SDXL Image Generation with LongCLIP encoders"
    )
    parser.add_argument(
        "--sdxl_ckpt",
        type=str,
        default="./ckpts/sdxl/sd_xl_base_1.0.safetensors",
        help="Path to the SDXL checkpoint",
    )
    parser.add_argument(
        "--coco_instances_json",
        type=str,
        default="./instances_val2014.json",
        help="Path to the COCO instances JSON",
    )
    parser.add_argument(
        "--total_images",
        type=int,
        default=10,
        help="Total number of images to generate",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./img_results/sdxl/",
        help="Output directory to save images",
    )
    parser.add_argument(
        "--seed", type=int, default=322, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch", type=int, default=3, help="Number of images per generation batch"
    )
    parser.add_argument(
        "--class_count",
        type=int,
        default=None,
        help="Number of distinct classes per image when using COCO",
    )
    parser.add_argument(
        "--use_coco_caption",
        action="store_true",
        help="Use COCO captions instead of prompt constructed by ourselves",
    )
    parser.add_argument(
        "--compose_sdxl",
        action="store_true",
        help="Use composable SDXL pipeline",
    )
    parser.add_argument(
        "--structure_guidance_sdxl",
        action="store_true",
        help="Use Structure Guidance SDXL pipeline",
    )
    parser.add_argument(
        "--layout_guidance_sdxl",
        action="store_true",
        help="Use Layout Guidance SDXL pipeline",
    )
    parser.add_argument(
        "--attend_and_excite_sdxl",
        action="store_true",
        help="Use Attend-and-Excite SDXL pipeline",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=30.0,
        help="Scale factor for Layout Guidance SDXL or Attend-and-Excite",
    )
    parser.add_argument(
        "--opt_layers",
        nargs=2,
        type=int,
        help="start end for the cross attention layers to optimize in Attend-and-Excite SDXL, e.g. --opt_layers 10 20 to optimize layers [10, 20), default is using all cross attention layers",
        default=None,
    )
    parser.add_argument(
        "--condComp",
        action="store_true",
        help="Use Sequential Inpainting pipeline",
    )
    parser.add_argument(
        "--decomposition_size",
        default=1,
        type=int,
        help="Number of objects are removed from the base stage and added later.",
    )
    parser.add_argument(
        "--keep_ratio_offset",
        type=int,
        default=4,
        help="Offset added to the keep-ratio denominator to reserve capacity for future composition stages.",
    )
    parser.add_argument(
        "--resample_steps",
        type=int,
        default=1,
        help="Number of resample steps",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.9,
        help="Strength for inpainting",
    )
    parser.add_argument(
        "--wo_layout_guidance",
        action="store_true",
        help="Disable layout guidance",
    )
    parser.add_argument(
        "--attnScore_decomp",
        action="store_true",
        help="Use attention score based decomposition",
    )
    parser.add_argument(
        "--objDetect_decomp",
        action="store_true",
        help="Use object detection based decomposition",
    )
    parser.add_argument(
        "--t_joint",
        type=float,
        default=0.4,
        help="Timestep for joint composition stage of attnScore decomposition",
    )
    parser.add_argument(
        "--t_comp",
        type=float,
        default=1.0,
        help="Timestep for composition stage of attnScore decomposition",
    )
    parser.add_argument(
        "--attn_threshold",
        type=float,
        default=0.5,
        help="Attention threshold for decomposition",
    )
    parser.add_argument(
        "--add_joint_composition_loss",
        action="store_true",
        help="Add joint composition loss",
    )
    parser.add_argument(
        "--ae_for_joint_stage",
        action="store_true",
        help="Use Attend-and-Excite for joint stage",
    )
    parser.add_argument(
        "--drop_stage_thres",
        type=float,
        default=1.0,
        help="Threshold for dropping stages",
    )

    args = parser.parse_args()

    main(
        args=args,
        total_images=args.total_images,
        out_dir=args.out_dir,
        seed=args.seed,
        batch=args.batch,
    )
