import torch
import torch.nn.functional as F
from RL2.datasets import pack_tensor_dicts
from RL2.utils.logging import time_logger
from RL2.utils.functions import aggregate_values

def compute_approx_kl(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    estimator: str
) -> torch.Tensor:
    # The (ref_)logps of non-action tokens are zero (see `Actor.
    # forward`), so their corresponding kl_term will also be zero.

    log_ratio = logps - ref_logps
    if estimator == "k1":
        return log_ratio
    elif estimator == "k2":
        return log_ratio.pow(2) / 2
    elif estimator == "k3":
        return log_ratio + torch.exp(- log_ratio) - 1
    else:
        raise NotImplementedError

def compute_gae(tensor_dict, gamma, lamda):
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = F.pad(tensor_dict["old_values"][:, 1:], (0, 1), value=0)
    deltas = tensor_dict["rewards"] + gamma * next_values - tensor_dict["old_values"]

    # A_t = \delta_t + \gamma * \lambda * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + tensor_dict["old_values"]

    action_gaes = gaes[torch.where(tensor_dict["action_mask"])]
    advantages = (gaes - action_gaes.mean()) * tensor_dict["action_mask"] / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    return {"advantages": advantages, "returns": returns}

def compute_reinforce_adv(
    tensor_dict,
    responses_per_prompt,
    global_norm: bool,
    norm_var: bool
):
    
    rewards = tensor_dict["rewards"].sum(-1).view(-1, responses_per_prompt)

    if global_norm:
        baseline = rewards.mean()
        std = rewards.std()
    else:
        baseline = rewards.mean(-1, keepdim=True)
        std = rewards.std(-1, keepdim=True)

    advantages = rewards - baseline
    if norm_var:
        advantages /= (
            std + torch.finfo(advantages.dtype).eps
        )

    advantages = advantages.view(-1, 1) * tensor_dict["action_mask"]
    return {"advantages": advantages}

@time_logger("compute_advantages")
def compute_advantages(
    config, tensor_dict, cu_seqs, step
):

    def extract_actions(tensor_dict):

        indices = torch.where(tensor_dict["action_mask"])
        return {
            k: v[indices]
            for k, v in tensor_dict.items()
        }
    
    processed_tensor_dict = pack_tensor_dicts([
        extract_actions(
            {
                k: v[start:end]
                for k, v in tensor_dict.items()
            }
        )
        for start, end in zip(cu_seqs[:-1], cu_seqs[1:])
    ])

    if config.estimator == "gae":
        tensor_dict_delta = compute_gae(
            processed_tensor_dict, config.gamma, config.lamda
        )
    elif config.estimator == "reinforce":
        tensor_dict_delta = compute_reinforce_adv(
            processed_tensor_dict,
            config.responses_per_prompt,
            config.global_norm,
            config.norm_var
        )
    else:
        raise NotImplementedError

    for k, v in tensor_dict_delta.items():
        tensor_dict[k] = torch.zeros(tensor_dict["states"].shape)
        for idx, (start, end) in enumerate(
            zip(cu_seqs[:-1], cu_seqs[1:])
        ):
            indices = torch.where(tensor_dict["action_mask"][start:end])
            tensor_dict[k][start:end][indices] = v[idx][:len(indices[0])]

def rm_loss(minibatch):

    chosen_rewards, rejected_rewards = minibatch["values"].sum(-1).view(-1, 2).T
    reward_margins = chosen_rewards - rejected_rewards
    losses = - F.logsigmoid(reward_margins)
    return losses, {"accuracy": (reward_margins > 0).tolist()}

def dpo_loss(config, minibatch):

    chosen_rewards, rejected_rewards = config.beta * (
        minibatch["logps"] - minibatch["ref_logps"]
    ).sum(-1).view(-1, 2).T
    reward_margins = chosen_rewards - rejected_rewards
    losses = - F.logsigmoid(reward_margins)
    metric = {
        "rewards/chosen": chosen_rewards.tolist(),
        "rewards/rejected": rejected_rewards.tolist(),
        "rewards/margin": reward_margins.tolist(),
        "accuracy": (reward_margins > 0).tolist()
    }
    return losses, metric

def actor_ppo_loss(config, minibatch):

    ratio = torch.exp(
        minibatch["logps"] - minibatch.get(
            "old_logps", minibatch["logps"].detach()
        )
    )
    clipped_ratio = torch.clamp(
        ratio, 1 - config.clip, 1 + config.clip
    )
    objective = minibatch["advantages"] * ratio
    clipped_objective = minibatch["advantages"] * clipped_ratio
    losses = - torch.min(objective, clipped_objective)
    clip_ratios = objective > clipped_objective

    if config.kl.coef > 0 and config.kl.type == "loss":
        kl_losses = compute_approx_kl(
            minibatch["logps"],
            minibatch["ref_logps"],
            config.kl.loss_estimator
        )
        losses = losses + config.kl.coef * kl_losses

    if config.tis_coef > 0:
        # https://fengyao.notion.site/off-policy-rl
        tis = torch.exp(
            minibatch["logps"].detach() - minibatch["llm_logps"]
        ).clamp(max=config.tis_coef)
        losses *= tis

    losses = losses - config.entropy.coef * minibatch["entropy"]
    return losses, clip_ratios

def critic_ppo_loss(config, minibatch):

    clipped_values = torch.clamp(
        minibatch["values"],
        minibatch["old_values"] - config.clip,
        minibatch["old_values"] + config.clip
    )
    mse = (minibatch["values"] - minibatch["returns"]).pow(2)
    clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
    losses = torch.max(mse, clipped_mse)
    clip_ratios = mse < clipped_mse
    return losses, clip_ratios

def actor_gspo_loss(config, minibatch, total_sequences):

    logps = minibatch["logps"]
    old_logps = minibatch.get("old_logps", logps.detach())
    advantages = minibatch["advantages"]
    action_mask = minibatch["action_mask"]

    seq_lengths = action_mask.sum(-1).clamp(min=1e-8)

    token_log_ratios = logps - old_logps
    sum_log_ratios = (token_log_ratios * action_mask).sum(-1)
    mean_log_ratios = sum_log_ratios / seq_lengths
    seq_ratios = torch.exp(mean_log_ratios)

    last_indices = (action_mask.sum(dim=1) - 1).long().clamp(min=0)
    seq_advantages = advantages[torch.arange(advantages.size(0)), last_indices]

    clipped_seq_ratios = torch.clamp(
        seq_ratios, 1 - config.clip, 1 + config.clip
    )
    objective = seq_ratios * seq_advantages
    clipped_objective = clipped_seq_ratios * seq_advantages
    seq_losses = - torch.min(objective, clipped_objective)

    loss = seq_losses.sum() / total_sequences
    clip_ratios = (objective > clipped_objective).float().sum() / total_sequences

    return loss, clip_ratios

def actor_cispo_loss(config, minibatch):

    clip_max = getattr(config, "clip_max", 5.0)

    logps = minibatch["logps"]
    old_logps = minibatch.get("old_logps", logps.detach())
    advantages = minibatch["advantages"]

    ratio = torch.exp(logps - old_logps)
    weights = torch.min(
        ratio, torch.tensor(clip_max, device=ratio.device)
    ).detach()

    pg_term = advantages * logps
    losses = - (weights * pg_term)

    return losses, weights

def track_tis_metrics(config, minibatch, total_actions, total_sequences):
    metrics = {}
    
    tis_raw = minibatch["logps"].detach() - minibatch["llm_logps"]
    tis_raw_v2 = 0.5 * (tis_raw ** 2)
    tis_raw_mean, tis_raw_v2_mean = aggregate_values(
        (tis_raw, tis_raw_v2),
        minibatch["action_mask"],
        config.avg_level,
        total_actions,
        total_sequences
    )
    metrics["actor/tis_raw_mean"] = tis_raw_mean.item()
    metrics["actor/tis_raw_v2_mean"] = tis_raw_v2_mean.item()
    
    return metrics