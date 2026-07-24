"""gpu_utils.py - GPU 메모리 관리 공용 유틸 (labeling/training 양쪽에서 사용)."""
import gc

import torch


def free_cuda():
    """무거운 학습/추론 직후 GPU 메모리가 곧바로 안 풀리는 경우가 있어 명시적으로 회수한다.

    (교훈) 대용량 조건에서 학습 직후 다음 단계가 OOM 으로 죽는 사고가 있었다.
    학습 -> 추론/변환으로 넘어가는 경계마다 호출한다.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
