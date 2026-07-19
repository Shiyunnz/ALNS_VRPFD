import csv
from pathlib import Path

from sensitivity.rerun_deadline_cost_sensitivity import (
    TRIAL_FIELDNAMES,
    TrialTask,
    append_trial_row,
    completed_task_keys,
    load_trial_rows,
    pending_tasks,
)


def test_append_trial_row_is_immediately_readable(tmp_path: Path) -> None:
    trial_csv = tmp_path / "trials.csv"
    task = TrialTask(
        analysis="gamma",
        instance="data/Instance10/R_30_10_1.txt",
        level_name="gamma",
        level_value=1,
        trial=0,
        seed=42,
    )

    append_trial_row(
        trial_csv,
        task,
        {
            "instance": task.instance,
            "gamma": 1,
            "best_cost": 12.5,
            "feasible": True,
            "run_time": 3.25,
        },
    )

    with trial_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == TRIAL_FIELDNAMES
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["analysis"] == "gamma"
    assert rows[0]["task_key"] == task.key()
    assert rows[0]["best_cost"] == "12.5"


def test_completed_task_keys_support_resume(tmp_path: Path) -> None:
    trial_csv = tmp_path / "trials.csv"
    done = TrialTask("battery", "data/Instance25/R_30_25_1.txt", "battery_capacity", 6.3, 2, 44)
    todo = TrialTask("battery", "data/Instance25/R_30_25_1.txt", "battery_capacity", 7.3, 2, 44)
    append_trial_row(trial_csv, done, {"instance": done.instance, "battery_capacity": 6.3})

    assert completed_task_keys(trial_csv) == {done.key()}
    assert pending_tasks([done, todo], completed_task_keys(trial_csv), force=False) == [todo]
    assert pending_tasks([done, todo], completed_task_keys(trial_csv), force=True) == [done, todo]


def test_load_trial_rows_converts_types_for_summary(tmp_path: Path) -> None:
    trial_csv = tmp_path / "trials.csv"
    task = TrialTask("timewindow", "data/Instance10/R_40_10_2.txt", "scale", 1.5, 1, 43)
    append_trial_row(
        trial_csv,
        task,
        {
            "instance": task.instance,
            "scale": 1.5,
            "best_cost": 88.0,
            "best_drone_customers": 3,
            "feasible": False,
            "cost_increase_vs_baseline": -2.5,
        },
    )

    rows = load_trial_rows(trial_csv, analysis="timewindow")

    assert rows == [
        {
            "instance": task.instance,
            "scale": 1.5,
            "best_cost": 88.0,
            "best_drone_customers": 3,
            "feasible": False,
            "cost_increase_vs_baseline": -2.5,
        }
    ]
