from .base import MegatronWorker

class MegatronActor(MegatronWorker):
    
    def __init__(self, config, train: bool):
        super().__init__(config, train)

        self.model = self.bridge.get_model()
        self.prepare_model_optimizer()