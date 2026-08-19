"""
Phase 1.2：参数统计与 FLOPs 估算（U2.3 核心练习：手算，不是数出来）
"""
import torch
from .config import ModelConfig


def count_params(cfg: ModelConfig) -> int:
    """
    闭式手算参数量：只用 config 里的数字推导，**不实例化模型**。

    验收等式：count_params(cfg) == sum(p.numel() for p in MiniLM(cfg).parameters())

    要数的块（清单给你，公式自己推）：
      - embedding
      - 每层：q/k/v/o 四个投影（注意哪些有 bias、哪些没有；
        GQA 下 k/v 的输出维度和 q 不同；head_dim 可能 != hidden//num_heads）
      - 每层：SwiGLU 的三个矩阵（全部无 bias）
      - 每层：两个 RMSNorm
      - 最后的 RMSNorm
      - lm_head —— 想想 weight tying 时它还算不算参数

    自查工具：先对一个 1 层小 config 手算，与 sum(p.numel()) 对不上时，
    逐 name 打印 model.named_parameters() 的 shape 找出漏数/多数的块。
    """
    raise NotImplementedError


def estimate_flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """
    估算单 token 前向 FLOPs：只计矩阵乘（一次乘加记 2 FLOPs），
    不含 softmax / norm 等逐元素操作。

    要数的矩阵乘（公式自己推）：
      - 每层：q/k/v/o 四个投影
      - 每层：attention 的两次 matmul（QK^T 和 AV）——
        这两项和 seq_len 有关，其余都无关，想清楚为什么
      - 每层：SwiGLU 三个矩阵
      - lm_head

    经验校验：seq_len 远小于 hidden 时，总 FLOPs 应约等于 2 * 参数量
    （每个参数每 token 参与一次乘加）。用这个关系自查。
    """
    raise NotImplementedError
