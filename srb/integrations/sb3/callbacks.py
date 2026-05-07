import numpy
from stable_baselines3.common.callbacks import BaseCallback


class RewardTermsTensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, list):
            return True

        reward_terms_per_env = [
            info.get("reward_terms")
            for info in infos
            if isinstance(info, dict) and isinstance(info.get("reward_terms"), dict)
        ]
        if not reward_terms_per_env:
            return True

        for reward_term in reward_terms_per_env[0].keys():
            reward_term_values: list[float] = []
            for reward_terms in reward_terms_per_env:
                value = reward_terms.get(reward_term)
                if value is None:
                    continue
                if isinstance(value, numpy.ndarray):
                    reward_term_values.append(float(value))
                elif hasattr(value, "item"):
                    reward_term_values.append(float(value.item()))
                else:
                    reward_term_values.append(float(value))

            if reward_term_values:
                self.logger.record_mean(
                    f"rollout/reward_terms/{reward_term}",
                    float(sum(reward_term_values) / len(reward_term_values)),
                )

        return True
