from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM
from RL2.workers import Worker
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.sequence_parallelism import sequence_parallelism_manager
from RL2.utils.functions import (
    compute_logsumexp,
    gather_action_logits,
    compute_entropy,
    aggregate_values
)
from RL2.utils.algorithms import compute_approx_kl
from RL2.utils.offloading import (
    init_weight_context,
    model_offloading_manager
)
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_reduce,
    rank0_log
)


class Actor(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)

        if config.use_liger_kernel:
            assert config.tp_size == 1, \
                "Liger kernel is not compatible with tensor parallelism."
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model_cls = AutoLigerKernelForCausalLM
        else:
            model_cls = AutoModelForCausalLM

        with init_weight_context(self):
            self.model = model_cls.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )

        self.prepare_model_optimizer()

    @sequence_parallelism_manager
    def forward(self, minibatch, return_entropy=False):

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.to(torch.float32) / getattr(
            self.config, "temperature", 1.0
        )
        # bfloat16 is unstable for the subsequent `logsumexp` operation.
        # See https://github.com/OpenRLHF/OpenRLHF/pull/634.
        
        logsumexp = compute_logsumexp(logits, self.device_mesh["tp"])
        action_logits = gather_action_logits(
            logits,
            minibatch["actions"],
            self.device_mesh["tp"]
        )
        logps = (action_logits - logsumexp) * minibatch["action_mask"]
        
        if return_entropy:
            entropy = compute_entropy(
                logits, logsumexp, self.device_mesh["tp"]
            ) * minibatch["action_mask"]
            return logps, entropy
        else:
            return logps

    @time_logger("compute_logps")
    @model_offloading_manager
    @torch.no_grad()
    @data_manager(gather=True)
    def compute_logps(self, minibatches, step):
        prefix = "old" if self.train else "ref"

        self.model.eval()
        for minibatch in progress_bar(
            minibatches, desc=f"Compute {prefix} logps"
        ):
            minibatch[f"{prefix}_logps"] = self.forward(minibatch)
        
        return minibatches
    
    @time_logger("update_actor")
    @model_offloading_manager
    @data_manager(pack_minibatches=True)
    def update(self, batches, step: int):
        if step < self.config.freeze_steps:
            return

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for batch in batches:
            
            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                self.device_mesh["dp"]
            )
            metric = defaultdict(list)
            for minibatch in batch:

                logps, entropy = self.forward(
                    minibatch, return_entropy=True
                )
                ratio = torch.exp(
                    logps - minibatch.get("old_logps", logps.detach())
                )
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip, 1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                losses = - torch.min(objective, clipped_objective)
                clip_ratios = objective > clipped_objective

                tis = None
                if self.config.tis_coef > 0:
                    # https://fengyao.notion.site/off-policy-rl
                    tis = torch.exp(
                        logps.detach() - minibatch["llm_logps"]
                    ).clamp(max=self.config.tis_coef)
                    losses *= tis

                # Track TIS metrics if enabled (independent of tis_coef)
                if getattr(self.config, 'track_tis', False):
                    tis_raw = logps.detach() - minibatch["llm_logps"]
                    tis_raw_v2 = 0.5 * (tis_raw ** 2)
                    tis_raw_mean = aggregate_values(
                        tis_raw,
                        minibatch["action_mask"],
                        self.config.avg_level,
                        total_actions,
                        total_sequences
                    )
                    tis_raw_v2_mean = aggregate_values(
                        tis_raw_v2,
                        minibatch["action_mask"],
                        self.config.avg_level,
                        total_actions,
                        total_sequences
                    )
                    metric["actor/tis_raw_mean"].append(tis_raw_mean.item())
                    metric["actor/tis_raw_v2_mean"].append(tis_raw_v2_mean.item())
                    
                    if self.config.tis_coef > 0 and tis is not None:
                        tis_clamped_mean = aggregate_values(
                            tis,
                            minibatch["action_mask"],
                            self.config.avg_level,
                            total_actions,
                            total_sequences
                        )
                        metric["actor/tis_clamped_mean"].append(tis_clamped_mean.item())
                    
                loss, clip_ratio, entropy = aggregate_values(
                    (losses, clip_ratios, entropy),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )
                loss = loss - self.config.entropy.coef * entropy
                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_approx_kl(
                        logps,
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                self.backward(loss)

                tbar.update()
                metric["actor/entropy"].append(entropy.item())
                metric["actor/loss"].append(loss.item())
                metric["actor/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()

            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"])
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
