"""Watch one Slurm array of training runs until it is done, or until something
needs a person. For documents/notes/ame_bank_sigma_grid.md.

usage:
  python scripts/watch_array.py HOST ARRAY_JOB_ID RUN_NAMES MAX_MINUTES [POLL_SECONDS]

RUN_NAMES is the comma list of the array's W&B run names (project ideas_cv/nano-ame),
one per task, in task order. Every poll prints one line per task: its Slurm state
and its W&B run, step and loss.

Exits, printing the reason on a line that starts with EXIT:
  - every task has ended (COMPLETED or not);
  - a task ended in any state but COMPLETED (failed, cancelled, timed out, ...);
  - a run's loss is NaN;
  - MAX_MINUTES passed;
  - with WATCH_START_STEP=N set: a task is running and every running task's W&B
    run has reached step N (the start check).

Task states come from sacct, which keeps a task after it leaves the queue, so an
ended task is never confused with an unreachable controller: a query that does not
exit 0 reads as unknown and is retried at the next poll. Poll slowly (default 600
s): bursts of ssh connections trip entropy's fail2ban.
"""

import math
import os
import signal
import subprocess
import sys
import time

import wandb

PROJECT = "ideas_cv/nano-ame"
START_STEP = int(os.environ["WATCH_START_STEP"]) if os.environ.get("WATCH_START_STEP") else None
WANDB_TIMEOUT = 120  # s for one W&B call; a hung call stalled a poll for 24+ min
TERMINAL_BAD = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED")


def task_states(host, array_id, n_tasks):
    """{task index: state}; tasks not started yet read PENDING. None if the
    query failed."""
    cmd = f"sacct -j {array_id} -X -n -P -o JobID,State; echo rc=$?"
    try:
        out = subprocess.run(
            ["timeout", "60", "ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
             host, cmd],
            capture_output=True, text=True, timeout=90,
        ).stdout
    except Exception:
        return None
    if "rc=0" not in out.split():
        return None
    states = {i: "PENDING" for i in range(n_tasks)}
    for line in out.splitlines():
        if "|" not in line:
            continue
        job, state = line.split("|", 1)
        task = job.split("_", 1)[1] if "_" in job else ""
        if task.isdigit() and int(task) < n_tasks:
            # "CANCELLED by 123" -> CANCELLED
            states[int(task)] = state.split()[0] if state else "UNKNOWN"
    return states


class Hung(BaseException):
    """A W&B call ran past WANDB_TIMEOUT. A BaseException, so wandb's own retry
    loops, which catch Exception, cannot swallow it."""


def _alarm(signum, frame):
    raise Hung()


def bounded(fn):
    """fn() or raise Hung after WANDB_TIMEOUT seconds."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WANDB_TIMEOUT)
    try:
        return fn()
    finally:
        signal.alarm(0)


def run_status(api, name):
    def lookup():
        runs = list(api.runs(PROJECT, filters={"display_name": name}, order="-created_at"))
        if not runs:
            return None
        run = runs[0]
        return run.id, run.summary.get("step"), run.summary.get("steps/train/loss")
    try:
        info = bounded(lookup)
    except Hung:
        return None, "wandb timed out"
    except Exception as exc:
        return None, f"wandb error {type(exc).__name__}"
    return (info, None) if info else (None, "no W&B run yet")


def main():
    host, array_id, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    deadline = time.time() + float(sys.argv[4]) * 60
    poll = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    while True:
        stamp = time.strftime("%m-%d %H:%M:%S")
        states = task_states(host, array_id, len(names))
        try:
            api, api_err = bounded(lambda: wandb.Api(timeout=60)), None
        except Hung:
            api, api_err = None, "wandb unavailable: timed out"
        except Exception as exc:
            api, api_err = None, f"wandb unavailable: {type(exc).__name__}"
        running_steps = []
        for i, name in enumerate(names):
            state = "?" if states is None else states[i]
            info, err = run_status(api, name) if api is not None else (None, api_err)
            rid, step, loss = info if info else ("-", None, None)
            print(f"[{stamp}] {array_id}_{i} {name} {state} W&B {rid} step {step} "
                  f"loss {loss} {err or ''}", flush=True)
            if isinstance(loss, float) and math.isnan(loss):
                print(f"EXIT: {name} ({array_id}_{i}) loss is NaN", flush=True)
                return
            if state.startswith(TERMINAL_BAD):
                print(f"EXIT: {name} ({array_id}_{i}) ended {state}", flush=True)
                return
            if state == "RUNNING":
                running_steps.append(step)
        if START_STEP is not None and running_steps and all(
                s is not None and s >= START_STEP for s in running_steps):
            print(f"EXIT: {len(running_steps)} running task(s) at step >= {START_STEP}",
                  flush=True)
            return
        if states is not None and all(s == "COMPLETED" for s in states.values()):
            print(f"EXIT: every task of {array_id} COMPLETED", flush=True)
            return
        if time.time() > deadline:
            print("EXIT: deadline", flush=True)
            return
        time.sleep(poll)


if __name__ == "__main__":
    main()
