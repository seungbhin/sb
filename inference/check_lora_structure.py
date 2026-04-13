import torch
import numpy as np
from diffusers import CogVideoXPipeline
from safetensors import safe_open

lora_path = "/home/sbjeon/workspace/my/train/output/dog_guitar_11_16/pytorch_lora_weights.safetensors"

print("=" * 80)
print("LoRA 가중치 파일 구조 분석")
print("=" * 80)

# 방법 1: safetensors로 직접 확인
print("\n[방법 1] safetensors 파일 직접 읽기:")
print("-" * 80)
try:
    with safe_open(lora_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"전체 key 개수: {len(keys)}")
        
        print("\n[전체 key 목록]")
        for i, k in enumerate(sorted(keys)):
            shape = f.get_tensor(k).shape
            print(f"{i:04d}: {k:80s} | shape: {shape}")
        
        print("\n[Key 패턴 분석]")
        patterns = {}
        for k in keys:
            # key 구조: transformer.transformer_blocks.{i}.attn1.to_q.lora_A.{adapter_name}
            parts = k.split('.')
            if len(parts) >= 2:
                pattern = '.'.join(parts[:2])  # transformer.transformer_blocks
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(k)
        
        for pattern in sorted(patterns.keys()):
            print(f"{pattern}: {len(patterns[pattern])} keys")
        
        print("\n[레이어 타입별 분석]")
        layer_types = {}
        for k in keys:
            if 'to_q' in k:
                layer_types.setdefault('to_q', []).append(k)
            elif 'to_k' in k:
                layer_types.setdefault('to_k', []).append(k)
            elif 'to_v' in k:
                layer_types.setdefault('to_v', []).append(k)
            elif 'to_out' in k:
                layer_types.setdefault('to_out', []).append(k)
            elif 'ff.net.0.proj' in k:
                layer_types.setdefault('ff.net.0.proj', []).append(k)
            elif 'ff.net.2' in k:
                layer_types.setdefault('ff.net.2', []).append(k)
        
        for layer_type in sorted(layer_types.keys()):
            print(f"{layer_type}: {len(layer_types[layer_type])} keys")
            # 처음 3개만 출력
            for k in sorted(layer_types[layer_type])[:3]:
                print(f"  - {k}")
            if len(layer_types[layer_type]) > 3:
                print(f"  ... and {len(layer_types[layer_type]) - 3} more")
        
except Exception as e:
    print(f"Error: {e}")

# 방법 2: CogVideoXPipeline.lora_state_dict로 확인
print("\n" + "=" * 80)
print("[방법 2] CogVideoXPipeline.lora_state_dict로 확인:")
print("-" * 80)
try:
    state_dict = CogVideoXPipeline.lora_state_dict(lora_path)
    print(f"전체 key 개수: {len(state_dict)}")
    
    print("\n[전체 key 목록]")
    for i, k in enumerate(sorted(state_dict.keys())):
        shape = state_dict[k].shape
        print(f"{i:04d}: {k:80s} | shape: {shape}")
    
    print("\n[레이어 타입별 분석]")
    layer_types = {}
    for k in state_dict.keys():
        if 'to_q' in k:
            layer_types.setdefault('to_q', []).append(k)
        elif 'to_k' in k:
            layer_types.setdefault('to_k', []).append(k)
        elif 'to_v' in k:
            layer_types.setdefault('to_v', []).append(k)
        elif 'to_out' in k:
            layer_types.setdefault('to_out', []).append(k)
        elif 'ff.net.0.proj' in k:
            layer_types.setdefault('ff.net.0.proj', []).append(k)
        elif 'ff.net.2' in k:
            layer_types.setdefault('ff.net.2', []).append(k)
    
    for layer_type in sorted(layer_types.keys()):
        print(f"\n{layer_type}: {len(layer_types[layer_type])} keys")
        # 처음 5개만 출력
        for k in sorted(layer_types[layer_type])[:5]:
            print(f"  - {k}")
        if len(layer_types[layer_type]) > 5:
            print(f"  ... and {len(layer_types[layer_type]) - 5} more")
    
    print("\n[Adapter 이름 분석]")
    adapter_names = set()
    for k in state_dict.keys():
        # key 구조에서 adapter 이름 추출
        # 예: transformer.transformer_blocks.0.attn1.to_q.lora_A.motion_adapter
        parts = k.split('.')
        if 'lora_A' in k or 'lora_B' in k:
            # adapter 이름은 보통 마지막 부분
            if len(parts) > 0:
                last_part = parts[-1]
                if last_part not in ['weight', 'bias']:
                    adapter_names.add(last_part)
    
    print(f"발견된 adapter 이름들: {sorted(adapter_names)}")
    
    # lora_A와 lora_B 키 개수 확인
    print("\n[LoRA 키 구조 분석]")
    print("-" * 80)
    lora_A_keys = [k for k in state_dict.keys() if 'lora_A' in k]
    lora_B_keys = [k for k in state_dict.keys() if 'lora_B' in k]
    print(f"lora_A 키 개수: {len(lora_A_keys)}")
    print(f"lora_B 키 개수: {len(lora_B_keys)}")
    
    if len(lora_A_keys) != len(lora_B_keys):
        print(f"⚠️ 경고: lora_A와 lora_B의 키 개수가 다릅니다!")
        print(f"   이것은 저장 과정에서 문제가 있었을 수 있습니다.")
    
    # lora_B가 실제로 0인지 몇 개 샘플 확인
    print(f"\n[lora_B 값 샘플 확인]")
    print("-" * 80)
    zero_count = 0
    non_zero_count = 0
    for key in sorted(lora_B_keys)[:10]:  # 처음 10개만
        tensor = state_dict[key]
        if torch.is_tensor(tensor):
            if tensor.dtype == torch.bfloat16:
                tensor_np = tensor.cpu().float().numpy()
            else:
                tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = np.array(tensor)
        
        is_zero = abs(tensor_np.mean()) < 1e-8 and abs(tensor_np.max()) < 1e-8
        if is_zero:
            zero_count += 1
            print(f"  {key}: 모두 0")
        else:
            non_zero_count += 1
            print(f"  {key}: 평균={tensor_np.mean():.6f}, 최대={tensor_np.max():.6f}")
    
    if zero_count > 0:
        print(f"\n⚠️ 경고: 확인한 lora_B 중 {zero_count}개가 모두 0입니다!")
        print(f"   이것이 LoRA가 작동하지 않는 원인일 수 있습니다.")
    
    # LoRA 값 적절성 검사
    print("\n" + "=" * 80)
    print("[LoRA 값 적절성 검사]")
    print("-" * 80)
    
    # 샘플 레이어 선택 (각 타입별로 1-2개씩, to_v 포함)
    sample_keys = []
    # 우선순위: to_v, to_q, to_k, to_out, ff.net 등
    priority_types = ['to_v', 'to_q', 'to_k', 'to_out', 'ff.net.0.proj', 'ff.net.2']
    
    for layer_type in priority_types:
        if layer_type in layer_types and layer_types[layer_type]:
            # lora_A와 lora_B 각각 하나씩 선택
            for k in sorted(layer_types[layer_type]):
                if 'lora_A' in k:
                    sample_keys.append(k)
                    break
            for k in sorted(layer_types[layer_type]):
                if 'lora_B' in k:
                    sample_keys.append(k)
                    break
    
    # 나머지 타입들도 추가 (to_v가 이미 포함되었는지 확인)
    for layer_type in sorted(layer_types.keys()):
        if layer_type not in priority_types and layer_types[layer_type]:
            # lora_A와 lora_B 각각 하나씩 선택
            for k in sorted(layer_types[layer_type]):
                if 'lora_A' in k:
                    sample_keys.append(k)
                    break
            for k in sorted(layer_types[layer_type]):
                if 'lora_B' in k:
                    sample_keys.append(k)
                    break
            if len(sample_keys) >= 12:  # 최대 12개 (to_v 포함)
                break
    
    print(f"\n샘플 레이어 {len(sample_keys)}개 확인:")
    print("-" * 80)
    
    for i, key in enumerate(sample_keys[:10], 1):  # 최대 10개만 출력
        tensor = state_dict[key]
        # BFloat16 등 특수 타입 처리
        if torch.is_tensor(tensor):
            # float32로 변환하여 numpy 변환
            if tensor.dtype == torch.bfloat16:
                tensor_np = tensor.cpu().float().numpy()
            else:
                tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = np.array(tensor)
            if tensor_np.dtype == np.dtype('bfloat16'):
                tensor_np = tensor_np.astype(np.float32)
        
        # 통계 정보
        mean_val = float(tensor_np.mean())
        std_val = float(tensor_np.std())
        min_val = float(tensor_np.min())
        max_val = float(tensor_np.max())
        
        # NaN/Inf 체크
        has_nan = bool(torch.isnan(tensor).any().item() if torch.is_tensor(tensor) else bool(np.isnan(tensor_np).any()))
        has_inf = bool(torch.isinf(tensor).any().item() if torch.is_tensor(tensor) else bool(np.isinf(tensor_np).any()))
        
        # lora_B가 모두 0인지 체크 (심각한 문제!)
        is_all_zero = bool(abs(max_val) < 1e-8 and abs(min_val) < 1e-8 and abs(mean_val) < 1e-8)
        
        # 적절성 판단
        is_normal = True
        warnings = []
        
        if has_nan:
            is_normal = False
            warnings.append("⚠️ NaN 발견!")
        if has_inf:
            is_normal = False
            warnings.append("⚠️ Inf 발견!")
        if is_all_zero and 'lora_B' in key:
            is_normal = False
            warnings.append("🚨 심각: lora_B가 모두 0입니다! LoRA가 작동하지 않습니다!")
        if abs(mean_val) > 10.0:
            warnings.append(f"⚠️ 평균값이 큼: {mean_val:.4f}")
        if std_val > 10.0:
            warnings.append(f"⚠️ 표준편차가 큼: {std_val:.4f}")
        if abs(max_val) > 100.0 or abs(min_val) > 100.0:
            warnings.append(f"⚠️ 값의 범위가 큼: [{min_val:.4f}, {max_val:.4f}]")
        
        if is_all_zero and 'lora_B' in key:
            status = "🚨 심각"
        elif is_normal and not warnings:
            status = "✅ 정상"
        else:
            status = "⚠️ 주의"
        
        print(f"\n[{i}] {key}")
        print(f"    Shape: {tensor.shape}")
        print(f"    평균: {mean_val:.6f} | 표준편차: {std_val:.6f}")
        print(f"    범위: [{min_val:.6f}, {max_val:.6f}]")
        print(f"    상태: {status}")
        if warnings:
            for w in warnings:
                print(f"    {w}")
    
    # 전체 통계 요약
    print("\n" + "-" * 80)
    print("[전체 LoRA 가중치 통계 요약]")
    print("-" * 80)
    
    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []
    nan_count = 0
    inf_count = 0
    zero_lora_B_count = 0  # lora_B가 모두 0인 레이어 개수
    
    for key, tensor in state_dict.items():
        if 'lora_A' in key or 'lora_B' in key:
            # BFloat16 등 특수 타입 처리
            if torch.is_tensor(tensor):
                # float32로 변환하여 numpy 변환
                if tensor.dtype == torch.bfloat16:
                    tensor_np = tensor.cpu().float().numpy()
                else:
                    tensor_np = tensor.cpu().numpy()
            else:
                tensor_np = np.array(tensor)
                if hasattr(tensor_np.dtype, 'name') and 'bfloat16' in str(tensor_np.dtype):
                    tensor_np = tensor_np.astype(np.float32)
            
            all_means.append(float(tensor_np.mean()))
            all_stds.append(float(tensor_np.std()))
            all_mins.append(float(tensor_np.min()))
            all_maxs.append(float(tensor_np.max()))
            
            # lora_B가 모두 0인지 체크
            if 'lora_B' in key:
                if abs(float(tensor_np.mean())) < 1e-8 and abs(float(tensor_np.max())) < 1e-8 and abs(float(tensor_np.min())) < 1e-8:
                    zero_lora_B_count += 1
            
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any():
                    nan_count += 1
                if torch.isinf(tensor).any():
                    inf_count += 1
            else:
                if np.isnan(tensor_np).any():
                    nan_count += 1
                if np.isinf(tensor_np).any():
                    inf_count += 1
    
    if all_means:
        print(f"전체 LoRA 레이어 수: {len(all_means)}")
        print(f"평균값 통계:")
        print(f"  - 전체 평균: {np.mean(all_means):.6f}")
        print(f"  - 평균 범위: [{np.min(all_means):.6f}, {np.max(all_means):.6f}]")
        print(f"표준편차 통계:")
        print(f"  - 전체 평균: {np.mean(all_stds):.6f}")
        print(f"  - 평균 범위: [{np.min(all_stds):.6f}, {np.max(all_stds):.6f}]")
        print(f"값 범위 통계:")
        print(f"  - 최소값 범위: [{np.min(all_mins):.6f}, {np.max(all_mins):.6f}]")
        print(f"  - 최대값 범위: [{np.min(all_maxs):.6f}, {np.max(all_maxs):.6f}]")
        
        if nan_count > 0:
            print(f"\n⚠️ 경고: NaN이 포함된 레이어: {nan_count}개")
        if inf_count > 0:
            print(f"⚠️ 경고: Inf가 포함된 레이어: {inf_count}개")
        
        # lora_B가 모두 0인 레이어 체크 (심각한 문제!)
        if zero_lora_B_count > 0:
            print(f"\n🚨 심각한 문제: lora_B가 모두 0인 레이어: {zero_lora_B_count}개")
            print(f"   LoRA는 lora_A @ lora_B로 작동하는데, lora_B가 0이면 LoRA 효과가 없습니다!")
            print(f"   가능한 원인:")
            print(f"   1. 학습이 제대로 되지 않았을 수 있음")
            print(f"   2. 저장할 때 문제가 있었을 수 있음")
            print(f"   3. LoRA 초기화가 잘못되었을 수 있음")
        else:
            print(f"\n✅ lora_B가 모두 0인 레이어 없음: LoRA가 정상적으로 작동할 수 있습니다.")
        
        if nan_count == 0 and inf_count == 0:
            print(f"✅ NaN/Inf 없음: 모든 레이어가 정상입니다.")
        
        # 적절성 판단
        if zero_lora_B_count > 0:
            print(f"\n🚨 LoRA가 제대로 작동하지 않을 수 있습니다! 학습/저장 과정을 확인하세요.")
        elif np.mean(np.abs(all_means)) < 1.0 and np.mean(all_stds) < 5.0:
            print(f"\n✅ LoRA 가중치가 적절한 범위에 있습니다.")
        else:
            print(f"\n⚠️ LoRA 가중치가 큰 범위에 있습니다. 학습이 제대로 되었는지 확인하세요.")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)

