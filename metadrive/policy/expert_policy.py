from metadrive.policy.env_input_policy import EnvInputPolicy


class ExpertPolicy(EnvInputPolicy):
    """
    This policy use built-in neural network to do control
    """

    def act(self, agent_id):
        vehicle = self.control_object
        from metadrive.examples.ppo_expert import expert
        try:
            saver_a, obs = expert(vehicle, deterministic=False, need_obs=True)
            obs = obs[0]
            return saver_a
        except:
            return [0, 0]
