from typing import Callable

from isaaclab.managers import (
    ActionManager as __ActionManager,
    ActionTerm,
    ActionTermCfg,
)



class ActionManager(__ActionManager):
    def _prepare_terms(self):
        # Create buffers to parse and store terms
        self._term_names: list[str] = []
        self._terms: dict[str, ActionTerm] = {}

        # Check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # Parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # Check if term config is None
            if term_cfg is None or isinstance(term_cfg, Callable):
                continue
            # Check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # Create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # Sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(
                    f"Returned object for the term '{term_name}' is not of type ActionType."
                )
            # Add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
