from relign.policies.actor_critic_policy import ActorCriticPolicy


class TestActorCritic:

    def test_save_checkpoint(self, actor_critic_policy: ActorCriticPolicy):
        actor_critic_policy._save_checkpoint("tests/mock_data/checkpoint")

    def test_save_latest_policy_path(self, actor_critic_policy: ActorCriticPolicy, experiment_dir):
        actor_critic_policy.init_actor_engine_if_needed()
        actor_critic_policy.init_critic_engine_if_needed()

        policy_path = actor_critic_policy.save_latest_policy_path()
        assert policy_path == f"{experiment_dir}/policy/cache/actor/hf_model"

    def test_load_latest_policy_path(self, actor_critic_policy: ActorCriticPolicy, experiment_dir):
        actor_critic_policy.init_actor_engine_if_needed()
        actor_critic_policy.init_critic_engine_if_needed()

        actor_critic_policy.load_latest_policy_path(project_root_dir=experiment_dir)

    def test_get_latest_checkpoint(self, actor_critic_policy: ActorCriticPolicy):
        latest_checkpoint = actor_critic_policy.get_latest_checkpoint()
        assert latest_checkpoint == "tests/mock_data/checkpoint"

    def test_load_checkpoint(self, actor_critic_policy: ActorCriticPolicy):
        actor_critic_policy.load_checkpoint("tests/mock_data/checkpoint")