from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class TrainingSnapshot:
    env_step: int
    total_env_steps: int
    episode: int
    episode_return: float
    episode_length: int
    epsilon: float
    replay_size: int
    replay_capacity: int
    gradient_steps: int
    latest_loss: Optional[float]

    mean_q_value: Optional[float] = None
    mean_target: Optional[float] = None

    greedy_actions: Optional[int] = None
    exploratory_actions: Optional[int] = None
    condition_triggers: Optional[int] = None
    steps_since_update: Optional[int] = None
    condition_triggered: Optional[bool] = None


class TrainingMonitor:
    def __init__(
        self,
        total_env_steps: int,
        experiment_name: str,
        device: str,
        refresh_per_second: int = 4,
        update_every_steps: int = 50,
    ) -> None:
        self.total_env_steps = total_env_steps
        self.experiment_name = experiment_name
        self.device = device
        self.update_every_steps = update_every_steps

        self.start_time = perf_counter()
        self.last_update_step = 0

        self.progress = Progress(
            TextColumn("[bold cyan]Training"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("{task.completed:,}/{task.total:,} env steps"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self.progress_task = self.progress.add_task(
            "training",
            total=total_env_steps,
        )

        self.live = Live(
            self._render_initial(),
            refresh_per_second=refresh_per_second,
            transient=False,
        )

    def __enter__(self) -> TrainingMonitor:
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.live.stop()

    @staticmethod
    def _format_float(value: Optional[float], digits: int = 4) -> str:
        if value is None:
            return "-"
        return f"{value:.{digits}f}"

    @staticmethod
    def _format_int(value: Optional[int]) -> str:
        if value is None:
            return "-"
        return f"{value:,}"

    def _render_initial(self):
        return Panel(
            Align.center("Preparing training..."),
            title=f"[bold green]{self.experiment_name}[/bold green]",
            border_style="green",
        )

    def _make_table(self, snapshot: TrainingSnapshot) -> Table:
        elapsed = perf_counter() - self.start_time
        steps_per_second = snapshot.env_step / elapsed if elapsed > 0 else 0.0

        table = Table(
            show_header=False,
            box=None,
            expand=True,
            padding=(0, 2),
        )
        table.add_column(style="bold cyan", width=28)
        table.add_column(style="white", justify="right")
        table.add_column(style="bold cyan", width=28)
        table.add_column(style="white", justify="right")

        table.add_row(
            "Experiment",
            self.experiment_name,
            "Device",
            self.device,
        )
        table.add_row(
            "Episode",
            f"{snapshot.episode:,}",
            "Current epsilon",
            f"{snapshot.epsilon:.4f}",
        )
        table.add_row(
            "Episode return",
            f"{snapshot.episode_return:.2f}",
            "Episode length",
            f"{snapshot.episode_length:,}",
        )
        table.add_row(
            "Replay buffer",
            f"{snapshot.replay_size:,}/{snapshot.replay_capacity:,}",
            "Gradient steps",
            f"{snapshot.gradient_steps:,}",
        )
        table.add_row(
            "Latest Huber loss",
            self._format_float(snapshot.latest_loss),
            "Steps / second",
            f"{steps_per_second:.1f}",
        )
        table.add_row(
            "Mean predicted Q",
            self._format_float(snapshot.mean_q_value),
            "Mean TD target",
            self._format_float(snapshot.mean_target),
        )

        if snapshot.greedy_actions is not None:
            table.add_row(
                "Greedy actions",
                self._format_int(snapshot.greedy_actions),
                "Exploratory actions",
                self._format_int(snapshot.exploratory_actions),
            )

        if snapshot.condition_triggers is not None:
            trigger_text = (
                "[bold green]YES[/bold green]"
                if snapshot.condition_triggered
                else "[dim]no[/dim]"
            )
            table.add_row(
                "Condition triggers",
                self._format_int(snapshot.condition_triggers),
                "Triggered this step",
                trigger_text,
            )
            table.add_row(
                "Steps since update",
                self._format_int(snapshot.steps_since_update),
                "",
                "",
            )

        return table

    def _render(self, snapshot: TrainingSnapshot):
        self.progress.update(
            self.progress_task,
            completed=snapshot.env_step,
        )

        title = (
            f"[bold green]{self.experiment_name}[/bold green] "
            f"[dim]| episode {snapshot.episode}[/dim]"
        )

        return Group(
            Panel(
                self.progress,
                border_style="cyan",
                title="[bold cyan]Training progress[/bold cyan]",
            ),
            Panel(
                self._make_table(snapshot),
                border_style="green",
                title=title,
            ),
        )

    def update(
        self,
        snapshot: TrainingSnapshot,
        force: bool = False,
    ) -> None:
        is_final_step = snapshot.env_step >= snapshot.total_env_steps
        should_update = (
            force
            or is_final_step
            or snapshot.env_step - self.last_update_step
            >= self.update_every_steps
        )

        if not should_update:
            return

        self.live.update(self._render(snapshot), refresh=True)
        self.last_update_step = snapshot.env_step