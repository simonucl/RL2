from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification
from RL2.workers.fsdp import FSDPWorker, whether_load_weight
from RL2.utils.sequences import count_total
from RL2.utils.fsdp.context_parallelism import context_parallelism_manager
from RL2.utils.functions import aggregate_values
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)


class FSDPCritic(FSDPWorker):

    def __init__(self, config):
        super().__init__(config, True)

        load_weight = whether_load_weight(self)
        kwargs = {
            "model_name_or_path": config.model_name,
            "num_labels": 1,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2"
        }
        if load_weight:
            self.model = AutoModelForTokenClassification.from_pretrained(**kwargs)
        else:
            with torch.device("meta"):
                self.model = AutoModelForTokenClassification.from_config(
                    config.model_name,
                )

        self.prepare_model_optimizer()

    @context_parallelism_manager
    def forward(self, minibatch) -> torch.Tensor:

        return self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.squeeze(-1) * minibatch["action_mask"]

    @time_logger("compute_values")
    @torch.no_grad()
    def compute_values(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)
        self.load_model_to_device(torch.cuda.current_device())

        self.model.eval()
        for minibatch in progress_bar(minibatches, desc="Compute values"):
            minibatch["values"] = self.forward(minibatch)

        self.load_model_to_device("cpu")
        return self.gather_data(minibatches)

    @time_logger("update_critic")
    def rm_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict, pair=True)

        total_pairs = count_total(
            minibatches, "eos_mask", self.device_mesh["dp"].get_group()
        ) // 2
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="Update critic"
        ):
            rewards = self.forward(minibatch)
            chosen_rewards, rejected_rewards = rewards.sum(-1).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / total_pairs
            self.backward(loss)
            metrics["loss"].append(loss.item())
            metrics["accuray"].extend((reward_margins > 0).tolist())

        grad_norm = self.optimizer_step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"])

    @time_logger("update_critic")
    def ppo_update(self, tensor_dict, step: int):
        batches = self.scatter_data(tensor_dict, pack_minibatches=True)
        self.load_model_to_device(torch.cuda.current_device())

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update critic"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                self.device_mesh["dp"].get_group()
            )
            metric = defaultdict(list)
            for minibatch in batch:

                values = self.forward(minibatch)
                clipped_values = torch.clamp(
                    values,
                    minibatch["values"] - self.config.clip,
                    minibatch["values"] + self.config.clip
                )
                mse = (values - minibatch["returns"]).pow(2)
                clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
                losses = torch.max(mse, clipped_mse)
                clip_ratios = mse < clipped_mse

                loss, clip_ratio = aggregate_values(
                    (losses, clip_ratios),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )

                self.backward(loss)

                tbar.update()
                metric["critic/loss"].append(loss.item())
                metric["critic/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()
            
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"])
                )
            metrics["critic/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        self.load_model_to_device("cpu")