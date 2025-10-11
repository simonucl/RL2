import hydra
import glob
import torch.distributed as dist
from tqdm import tqdm
import wandb
from RL2.trainer import Trainer
from RL2.datasets import RLDataset, get_dataloader
from RL2.workers import (
    initialize_actor,
    initialize_critic,
    initialize_rollout
)
from RL2.utils.commication import initialize_global_process_group
from RL2.utils.algorithms import (
    compute_approx_kl, compute_advantages
)
from RL2.utils.sglang import make_request
from RL2.utils.logging import time_logger


class PPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        self.train_dataloader = self.get_dataloader(True)
        self.test_dataloader = self.get_dataloader(False)
        self.actor.prepare_scheduler(
            self.config.trainer.n_epochs * len(self.train_dataloader)
        )
        if config.actor.kl.coef > 0:
            self.ref_actor = initialize_actor(config.ref_actor, False)
        if config.adv.estimator == "gae":
            self.critic = initialize_critic(config.critic)
            self.critic.prepare_scheduler(
                self.config.trainer.n_epochs * len(self.train_dataloader)
            )
        else:
            self.critic = None
        self.rollout = initialize_rollout(config.rollout)    

    def get_dataloader(self, train: bool):

        dataset = RLDataset(
            self.config.train_data
            if train else self.config.test_data,
            self.actor.tokenizer
        )

        return get_dataloader(
            dataset,
            self.config.train_data.prompts_per_rollout
            if train else len(dataset)
        )

    def load_ckpt(self):

        save_dir = self.config.trainer.load_ckpt_from
        if save_dir is None or (save_dir == "latest" and not glob.glob(f"{self.config.trainer.save_dir}/step*")):
            return 0
        if self.rollout.device_mesh["tp"].get_local_rank() == 0:
            make_request(
                self.rollout.worker_url, "release_memory_occupation"
            )
        step = super().load_ckpt((self.actor, self.critic))
        self.rollout.update(self.actor, step)
        return step
    
    @time_logger("compute_approx_kl")
    def compute_approx_kl(self, tensor_dict, step):

        approx_kl = compute_approx_kl(
            tensor_dict["old_logps"],
            tensor_dict["ref_logps"],
            self.config.actor.kl.reward_estimator
        )
        if self.config.actor.kl.type == "reward":
            tensor_dict["rewards"] -= self.config.actor.kl.coef * approx_kl
        wandb.log({
            "actor/kl": (approx_kl.sum() / tensor_dict["action_mask"].sum()).item()
        }, step=step)
            
    def train(self):

        step = self.load_ckpt()
        for epoch in range(
            step // len(self.train_dataloader),
            self.config.trainer.n_epochs
        ):
            for data_list in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1

                tensor_dict, cu_seqs = self.rollout(data_list, True, step)

                if self.config.actor.kl.coef > 0 or self.config.actor.update_per_rollout > 1:
                    tensor_dict = self.actor.compute_logps(tensor_dict, step)
                if self.config.actor.kl.coef > 0:
                    tensor_dict = self.ref_actor.compute_logps(tensor_dict, step)
                if self.config.adv.estimator == "gae":
                    tensor_dict = self.critic.compute_values(tensor_dict, step)

                if dist.get_rank() == 0:
                    if self.config.actor.kl.coef > 0:
                        self.compute_approx_kl(tensor_dict, step)
                    compute_advantages(self.config.adv, tensor_dict, cu_seqs, step)

                self.actor.ppo_update(tensor_dict, step)
                if self.config.adv.estimator == "gae":
                    self.critic.ppo_update(tensor_dict, step)
                self.save_ckpt(
                    (self.actor, self.critic), step
                )

                self.rollout.update(self.actor, step)
                if self.config.trainer.test_freq is not None and step % self.config.trainer.test_freq == 0:
                    for data_list in self.test_dataloader:
                        self.rollout(data_list, False, step)

        self.save_model((self.actor, self.critic))


@hydra.main(config_path="config", config_name="ppo", version_base=None)
def main(config):

    initialize_global_process_group()
    
    trainer = PPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()