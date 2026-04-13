import torch
import random
import numpy as np
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import argparse

def is_adapter_empty(pipe, adapter_name):
    count = 0
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
            count += 1
        if hasattr(module, "lora_B") and adapter_name in module.lora_B:
            count += 1
    return count == 0, count

def check_adapter_scales(pipe, adapter_names):
    """각 adapter의 scale이 실제로 적용되었는지 확인"""
    from peft.tuners.tuners_utils import BaseTunerLayer
    
    print("\n=== Adapter Scale 확인 ===")
    for adapter_name in adapter_names:
        scales = []
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "scaling") and adapter_name in module.scaling:
                    scale = module.scaling[adapter_name].item() if hasattr(module.scaling[adapter_name], "item") else module.scaling[adapter_name]
                    scales.append((name, scale))
                elif hasattr(module, "get_scale"):
                    try:
                        scale = module.get_scale(adapter_name)
                        if scale is not None:
                            scale_val = scale.item() if hasattr(scale, "item") else scale
                            scales.append((name, scale_val))
                    except:
                        pass
        
        if scales:
            print(f"{adapter_name}: {len(scales)}개 레이어에서 발견")
            # 처음 3개만 출력
            for name, scale in scales[:3]:
                print(f"  {name}: scale={scale}")
            if len(scales) > 3:
                print(f"  ... 외 {len(scales)-3}개 레이어")
        else:
            print(f"{adapter_name}: scale이 설정되지 않음!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument(
        "--id_scale", type=int, default=1,
    )
    parser.add_argument(
        "--motion_scale", type=int, default=1,
    )    
    parser.add_argument(
        "--shared_scale", type=int, default=1,
    )
    args = parser.parse_args()

    seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe = CogVideoXPipeline.from_pretrained(
        '/home/sbjeon/workspace/DualReal/CogVideoX-5b',
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    lora_path = "/home/sbjeon/workspace/my/train/output/dog_guitar_20260125_lora_8/pytorch_lora_weights.safetensors"
    # Lora rank shared=8, etc=16 with k_adapter, v_adapter, shared_adapter위에께 이거임. 
    #  "/home/sbjeon/workspace/my/train/output/dog_guitar copy/pytorch_lora_weights.safetensors"
    # "/home/sbjeon/workspace/my/train/output/dog_guitar copy/pytorch_lora_weights.safetensors"
    full_state_dict = CogVideoXPipeline.lora_state_dict(lora_path)

    id_keys = {k: v for k, v in full_state_dict.items() if "to_k" in k or "to_q" in k}
    pipe.load_lora_weights(id_keys, adapter_name="id_adapter")

    motion_keys = {k: v for k, v in full_state_dict.items() if any(t in k for t in ["to_v", "ff.net.2.proj", "to_out.0"])}
    pipe.load_lora_weights(motion_keys, adapter_name="motion_adapter")
        
    shared_keys = {
        k: v for k, v in full_state_dict.items()
        if any(x in k for x in ["ff.net.0.proj"])
    }
    pipe.load_lora_weights(shared_keys, adapter_name="shared_adapter") 

    ## lora layer 확인
    # print("[ID Adapter] Keys:")
    # for i, k in enumerate(sorted(id_keys.keys())):
    #     print(f"{i:03d}: {k}")

    # print("[Motion Adapter] Keys:")
    # for i, k in enumerate(sorted(motion_keys.keys())):
    #     print(f"{i:03d}: {k}")

    # print("[Shared Adapter] Keys:")
    # for i, k in enumerate(sorted(shared_keys.keys())):
    #     print(f"{i:03d}: {k}")

    print("\n=== 로드된 Adapter 목록 ===")
    list_adapters = pipe.get_list_adapters()
    print(f"List adapters: {list_adapters}")
    
    for adapter_name in ["id_adapter", "motion_adapter", "shared_adapter"]:
        empty, count = is_adapter_empty(pipe, adapter_name)
        print(f"{adapter_name}: empty={empty}, count={count}")

    print("\n=== 활성화된 Adapter ===")
    active_adapters = pipe.get_active_adapters()
    print(f"Active adapters: {active_adapters}")
    
    check_adapter_scales(pipe, ["id_adapter", "motion_adapter", "shared_adapter"])
    

    print("\n=== Inference 시작 ===")
    
    prompt = "A dog wearing a knitted sweater in front of a cozy fireplace cabin is playing guitar"
    # "A dog wearing a knitted sweater in front of a cozy fireplace cabin is playing guitar"
    # "A dog wearing a knitted sweater in front of a cozy fireplace cabin is playing guitar"
    # "A dog wearing round sunglasses on a sandy beach is playing guitar"

    id_scale = args.id_scale
    motion_scale = args.motion_scale
    shared_scale = args.shared_scale

    pipe.set_adapters(["id_adapter", "motion_adapter", "shared_adapter"], [id_scale, motion_scale, shared_scale])

    frames = pipe(
        prompt,
        guidance_scale=6,
        negative_prompt="", # default 
        use_dynamic_cfg=False,
        generator=generator,  # 같은 seed
        # attention_kwargs={"scale": 1.0},
    ).frames[0]
    
    first_frame = frames[0]
    first_frame.save(
        f"output_scale_{id_scale}_{motion_scale}_{shared_scale}_sweater_8_guidance_6_first_frame.png"
    )

    export_to_video(frames, f"output_scale_{id_scale}_{motion_scale}_{shared_scale}_sweater_8_guidance_6.mp4", fps=8)
    