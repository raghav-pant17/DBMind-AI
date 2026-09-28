"""Evaluation metrics."""

from dataclasses import dataclass


# Stores aggregate evaluation metrics.
@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    executable: int
    table_selection_matches: int
    unsafe_blocked: int

    # Calculates the share of cases that produced executable results.
    @property
    def execution_accuracy(self) -> float:
        return self.executable / self.total if self.total else 0.0

    # Calculates the share of cases selecting the expected tables.
    @property
    def correct_table_selection(self) -> float:
        return self.table_selection_matches / self.total if self.total else 0.0

    # Calculates the share of unsafe cases blocked before execution.
    @property
    def unsafe_block_rate(self) -> float:
        return self.unsafe_blocked / self.total if self.total else 0.0
