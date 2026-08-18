from isaaclab.app import AppLauncher as _IsaacLabAppLauncher


class AppLauncher(_IsaacLabAppLauncher):
    def _hide_stop_button(self):
        if self._livestream >= 1 or not self._headless:
            import omni.kit.widget.toolbar

            toolbar = omni.kit.widget.toolbar.get_instance()
            play_button_group = getattr(
                getattr(toolbar, "_builtin_tools", None), "_play_button_group", None
            )
            stop_button = getattr(play_button_group, "_stop_button", None)
            if stop_button is not None:
                stop_button.visible = False
                stop_button.enabled = False
                play_button_group._stop_button = None

    def _hide_play_button(self, flag):
        if self._livestream >= 1 or not self._headless:
            import omni.kit.widget.toolbar

            toolbar = omni.kit.widget.toolbar.get_instance()
            play_button_group = getattr(
                getattr(toolbar, "_builtin_tools", None), "_play_button_group", None
            )
            play_button = getattr(play_button_group, "_play_button", None)
            if play_button is not None:
                play_button.visible = not flag
                play_button.enabled = not flag
