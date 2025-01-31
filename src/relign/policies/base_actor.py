from relign.policies.base_policy import BasePolicy

from relign.policies.base_policy import ForwardOutput

class ActorPolicy(BasePolicy):
    def __init__(self, actor_model_fn, actor_config):
        self.actor_model_fn = actor_model_fn
        self.actor_config = actor_config
    
    def init_actor_model(self):
        return self.actor_model_fn()

    def get_actor_model(self):
        return self.actor_model_fn()

    def get_actor_config(self):
        return self.actor_config
    
    def forward() -> ForwardOutput:
        pass

    def backward():
        pass
