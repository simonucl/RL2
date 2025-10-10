from mbridge.utils.post_creation_callbacks import make_value_model
from .base import MegatronWorker

class MegatronCritic(MegatronWorker):
    
    def __init__(self, config):
        super().__init__(config, True)

        self.model = self.bridge.get_model(
            wrap_with_ddp=True,
            post_model_creation_callbacks=[
                make_value_model
            ]
        )
        self.prepare_model_optimizer()