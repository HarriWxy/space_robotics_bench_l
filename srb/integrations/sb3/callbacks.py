import numpy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class RewardTermsTensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._tensorboard_writer = None

    def _on_training_start(self) -> None:
        self._tensorboard_writer = None
        for output_format in getattr(self.logger, "output_formats", []):
            if isinstance(output_format, TensorBoardOutputFormat):
                self._tensorboard_writer = output_format.writer
                break

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, list):
            return True

        reward_terms_by_name: dict[str, list[float]] = {}
        task_metrics_by_name: dict[str, list[float]] = {}
        for info in infos:
            if not isinstance(info, dict):
                continue

            reward_terms = info.get("reward_terms")
            if isinstance(reward_terms, dict):
                for reward_term, value in reward_terms.items():
                    reward_terms_by_name.setdefault(reward_term, []).extend(
                        self._to_float_list(value)
                    )

            for name, value in info.items():
                if isinstance(name, str) and name.startswith("metrics/"):
                    metric_name = name.removeprefix("metrics/")
                    task_metrics_by_name.setdefault(metric_name, []).extend(
                        self._to_float_list(value)
                    )

        if not reward_terms_by_name and not task_metrics_by_name:
            return True

        for reward_term, reward_term_values in sorted(reward_terms_by_name.items()):
            if not reward_term_values:
                continue

            reward_term_array = numpy.asarray(reward_term_values, dtype=numpy.float32)
            mean_reward_term = float(reward_term_array.mean())

            if self._tensorboard_writer is not None:
                self._tensorboard_writer.add_scalar(
                    f"rollout/reward_terms/{reward_term}",
                    mean_reward_term,
                    self.num_timesteps,
                )
                # self._tensorboard_writer.add_histogram(
                #     f"rollout/reward_terms/{reward_term}/distribution",
                #     reward_term_array,
                #     self.num_timesteps,
                # )
                # self._tensorboard_writer.add_scalar(
                #     f"rollout/reward_terms/{reward_term}/min",
                #     float(reward_term_array.min()),
                #     self.num_timesteps,
                # )
                # self._tensorboard_writer.add_scalar(
                #     f"rollout/reward_terms/{reward_term}/max",
                #     float(reward_term_array.max()),
                #     self.num_timesteps,
                # )
                # self._tensorboard_writer.add_scalar(
                #     f"rollout/reward_terms/{reward_term}/std",
                #     float(reward_term_array.std()),
                #     self.num_timesteps,
                # )

            self.logger.record_mean(
                f"rollout/reward_terms/{reward_term}",
                mean_reward_term,
                exclude="tensorboard" if self._tensorboard_writer is not None else None,
            )

        for metric_name, metric_values in sorted(task_metrics_by_name.items()):
            if not metric_values:
                continue
            metric_array = numpy.asarray(metric_values, dtype=numpy.float32)
            mean_metric = float(metric_array.mean())
            tag = f"rollout/metrics/{metric_name}"
            if self._tensorboard_writer is not None:
                self._tensorboard_writer.add_scalar(
                    tag,
                    mean_metric,
                    self.num_timesteps,
                )
            self.logger.record_mean(
                tag,
                mean_metric,
                exclude="tensorboard" if self._tensorboard_writer is not None else None,
            )

        return True

    @staticmethod
    def _to_float_list(value: object) -> list[float]:
        if isinstance(value, numpy.ndarray):
            return numpy.asarray(value, dtype=numpy.float32).reshape(-1).tolist()

        if hasattr(value, "detach") and hasattr(value, "cpu"):
            tensor = value.detach().cpu()
            return numpy.asarray(tensor, dtype=numpy.float32).reshape(-1).tolist()

        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]

        if hasattr(value, "item"):
            return [float(value.item())]

        return [float(value)]
