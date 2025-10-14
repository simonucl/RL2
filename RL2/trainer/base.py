from omegaconf import OmegaConf
import glob
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config

        if dist.get_rank() == 0:
            print(OmegaConf.to_yaml(config))
            if config.trainer.use_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
            else:
                wandb.log = lambda *args, **kwargs: None

    def get_ckpt(self, step):
        return {
            "step": step,
            "dataloader": self.train_dataloader.state_dict()
        }

    def load_ckpt(self, workers):

        save_dir = self.config.trainer.load_ckpt_from
        if save_dir is None:
            return 0
        if save_dir == "latest":
            save_dirs = glob.glob(f"{self.config.trainer.save_dir}/step*")
            if not save_dirs:
                return 0
            save_dir = max(
                save_dirs, key=lambda dir: int(dir.split("/step")[-1])
            )
        
        for worker in workers:
            worker.load_ckpt(f"{save_dir}/{worker.__class__.__name__.lower()}")

        ckpt = self.get_ckpt(0)
        dcp.load(ckpt, checkpoint_id=f"{save_dir}/trainer")
        self.train_dataloader.load_state_dict(ckpt["dataloader"])
        return ckpt["step"]

    def save_ckpt(self, workers, step):

        if self.config.trainer.save_freq is None or step % self.config.trainer.save_freq != 0:
            return

        save_dir = f"{self.config.trainer.save_dir}/step{step}"
        for worker in workers:
            worker.save_ckpt(f"{save_dir}/{worker.__class__.__name__.lower()}")

        dcp.save(
            self.get_ckpt(step),
            checkpoint_id=f"{save_dir}/trainer"
        )

    def save_model(self, workers):

        save_dir = self.config.trainer.save_dir
        if self.config.trainer.save_freq is not None:
            save_dir += "/latest"
        
        for worker in workers:
            worker.save_model(save_dir)