"""参考答案包 5.2-grpo。"""
from .reward import parse_answer, reward_fn
from .advantage import group_advantages
from .loss import grpo_loss
from .rollout import rollout

__all__ = ["parse_answer", "reward_fn", "group_advantages", "grpo_loss", "rollout"]
