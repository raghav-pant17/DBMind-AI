"""Golden dataset evaluator."""

import json
from pathlib import Path

from dbmind_ai.evaluation.metrics import EvaluationMetrics
from dbmind_ai.graph.workflow import DBMindWorkflow


# Evaluates workflow behavior against a golden dataset.
class Evaluator:
    """Evaluate DBMind-AI against a small golden dataset."""

    # Stores the workflow and dataset path for evaluation.
    def __init__(self, workflow: DBMindWorkflow, dataset_path: Path) -> None:
        self.workflow = workflow
        self.dataset_path = dataset_path

    # Runs all evaluation cases and returns aggregate metrics.
    def run(self) -> EvaluationMetrics:
        cases = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        executable = 0
        table_selection_matches = 0
        unsafe_blocked = 0

        for case in cases:
            response = self.workflow.run(case["question"])
            executable += int(response.result is not None)
            table_selection_matches += int(
                all(table in response.sql for table in case["expected_tables"])
            )
            if case.get("unsafe") and response.result is None:
                unsafe_blocked += 1

        return EvaluationMetrics(
            total=len(cases),
            executable=executable,
            table_selection_matches=table_selection_matches,
            unsafe_blocked=unsafe_blocked,
        )
