import hydra
from collections import defaultdict
import torch.distributed as dist
from tqdm import tqdm
import wandb
from RL2.trainer import Trainer
from RL2.datasets import OfflineRLDataset, get_dataloader
from RL2.workers import Actor
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.functions import aggregate_values
from RL2.utils.algorithms import compute_offline_advantages, compute_approx_kl
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.checkpointing import load_ckpt, save_ckpt, save_model
from RL2.utils.logging import progress_bar, time_logger, gather_and_log

@time_logger("update_actor")
@data_manager()
def update(worker, minibatches, step):

    total_actions, total_sequences = count_total(
        minibatches,
        ("action_mask", "eos_mask"),
        worker.device_mesh["dp"]
    )
    metrics = defaultdict(list)
    for minibatch in progress_bar(
        minibatches, desc="Update actor"
    ):
        # Remove non-sequence data before forward pass
        forward_minibatch = {
            k: v for k, v in minibatch.items() 
            if k not in ["advantages", "labels", "kl_penalty"]
        }
        logps = worker.forward(forward_minibatch)
        
        # Compute base policy gradient loss
        policy_loss = - logps * minibatch["advantages"]
        
        # Add KL penalty if available
        if "kl_penalty" in minibatch:
            policy_loss += minibatch["kl_penalty"]
        
        loss = aggregate_values(
            policy_loss,
            minibatch["action_mask"],
            worker.config.avg_level,
            total_actions,
            total_sequences
        )
        worker.backward(loss)
        metrics["loss"].append(loss.item())

    grad_norm = worker.optimizer_step()
    metrics["grad_norm"].append(grad_norm)
    gather_and_log(metrics, worker.device_mesh["dp"], step)


class OfflineRLTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, True)
        dataset = OfflineRLDataset(
            config.data, self.actor.tokenizer
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.actor.scheduler = self.prepare_scheduler(self.actor)
        
        # Add reference actor for KL regularization
        if config.offline_rl.kl_coef > 0:
            self.ref_actor = Actor(config.ref_actor, False)

    @time_logger("compute_advantages")
    def compute_advantages(self, tensor_dict, step):
        
        tensor_dict['advantages'] = compute_offline_advantages(
            tensor_dict,
            tensor_dict["labels"],
            self.config.offline_rl.positive_label_scale,
            self.config.offline_rl.negative_label_scale,
            self.config.offline_rl.norm_var
        )["advantages"]

    @time_logger("compute_approx_kl")
    def compute_approx_kl(self, tensor_dict, step):

        approx_kl = compute_approx_kl(
            tensor_dict["old_logps"],
            tensor_dict["ref_logps"],
            self.config.offline_rl.kl_estimator
        )
        tensor_dict["kl_penalty"] = self.config.offline_rl.kl_coef * approx_kl
        wandb.log({
            "actor/kl": (approx_kl.sum() / tensor_dict["action_mask"].sum()).item()
        }, step=step)

    def update_reference_actor(self):
        """Update reference actor with current actor's state at the end of epoch"""
        if hasattr(self, 'ref_actor'):
            self.ref_actor.model.load_state_dict(self.actor.model.state_dict())
            if dist.get_rank() == 0:
                print("Updated reference actor with current actor state")

    def train(self):

        step = load_ckpt(self, (self.actor,))
        for epoch in range(
            step // len(self.train_dataloader),
            self.config.trainer.n_epochs
        ):
            for tensor_dict in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1
                
                # Compute log probabilities for KL if needed
                if self.config.offline_rl.kl_coef > 0:
                    # Filter out non-sequence tensors before compute_logps
                    logps_tensor_dict = {
                        k: v for k, v in tensor_dict.items() 
                        if k not in ["labels"]
                    }
                    logps_tensor_dict = self.actor.compute_logps(logps_tensor_dict, step)
                    logps_tensor_dict = self.ref_actor.compute_logps(logps_tensor_dict, step)
                    # Merge back the logps results
                    tensor_dict.update(logps_tensor_dict)
                
                if dist.get_rank() == 0:
                    self.compute_advantages(tensor_dict, step)
                    if self.config.offline_rl.kl_coef > 0:
                        self.compute_approx_kl(tensor_dict, step)
                
                update(self.actor, tensor_dict, step)
                save_ckpt(self, (self.actor,), step)
            
            # Update reference actor at the end of each epoch
            if self.config.offline_rl.kl_coef > 0:
                self.update_reference_actor()
                
        save_model(self, self.actor)


@hydra.main(config_path="config", config_name="offline_rl", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = OfflineRLTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()