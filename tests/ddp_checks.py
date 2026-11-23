"""Spawnable DDP fixtures and fault controls used by the resource tests."""

from pathlib import Path
import json
import os
import random

from nn_window_ablation import ddp_launch, execution

from tests.parallel_evaluation_checks import await_file


def toy_rank(config, group, _args, _started, _startup):
    import torch

    from nn_window_ablation.distributed_training import DistributedTrainer
    from nn_window_ablation.training import UpdateBudget
    from tests.p4_checks import Scalar, convex_loss

    with execution._training_stream(config) as stream:
        model = Scalar()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.125, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
        trainer = DistributedTrainer(
            model, optimizer, scheduler, stream, UpdateBudget(1, 6, 2), architecture={"fixture": "scalar"},
            group=group,
        )
        trainer.attach()
        seen = []

        def counted(model, shots, cores, buffers):
            seen.append(len(cores))
            assert model is trainer.ddp
            return convex_loss(model, shots, cores, buffers)

        trace = trainer.step(counted)
        (config.paths.run_root / f"rank{group.rank}.json").write_text(json.dumps({
            "gradient": model.theta.grad.item(), "theta": model.theta.item(), "scheduler": scheduler.state_dict(),
            "lr": optimizer.param_groups[0]["lr"], "microbatches": seen,
            "cores": trace.cores, "counters": trainer.counters,
        }))
        trainer.detach()
        group.barrier()


def capture_gradients(directory, rank):
    import torch

    from nn_window_ablation.training import Trainer

    previous = Trainer._optimize

    def record(self, *args, **kwargs):
        result = previous(self, *args, **kwargs)
        torch.save({
            "gradients": {name: value.grad for name, value in self.model.named_parameters()},
            "weights": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, directory / f"check-rank{rank}-update{self.phase_updates + 1}.pt")
        return result

    return record


def capture_rank(config, group, args, started, startup):
    from nn_window_ablation.training import Trainer

    Trainer._optimize = capture_gradients(config.paths.run_root / config.run_name, group.rank)
    execution._run_training(config, group=group, started=started, startup=startup, **args)


def recovery_rank(config, group, args, started, startup):
    import torch

    from nn_window_ablation.distributed_training import DistributedTrainer
    from nn_window_ablation.training import MemoryObjective

    original = MemoryObjective.__call__

    def exercise_model_rng(self, model, shots, cores, buffers):
        # Draws depend on supplied rows, so ranks consume different quantities.
        for _ in range(1 + (int(shots.shot_ids[0]) & 0xFFFFFFFF) % 7):
            torch.rand(())
            random.random()
        if os.environ.get("NNWA_DDP_TEST_FAULT") == "exception" and group.rank == 1:
            raise ValueError("rank 1 injected failure before backward")
        return original(self, model, shots, cores, buffers)

    previous = DistributedTrainer.step

    def step(self, objective):
        trace = previous(self, objective)
        if os.environ.get("NNWA_DDP_TEST_FAULT") == "pause" and trace.update == 3 and group.rank == 1:
            barrier = config.paths.run_root.parent / "rank1-ready"
            barrier.write_text(str(os.getpid()))
            await_file(barrier.with_name("rank1-release"))
        return trace

    MemoryObjective.__call__ = exercise_model_rng
    DistributedTrainer.step = step
    execution._run_training(config, group=group, started=started, startup=startup, **args)


def timeout_rank(config, group, _args, _started, _startup):
    base = config.paths.run_root
    ready = base / "timeout-ready"
    if group.owner:
        await_file(ready)
        group.barrier()
    else:
        ready.touch()
        await_file(base / "timeout-release")


def failing_exit_rank(config, group, args, started, startup):
    close = group.close

    def fail_after_teardown():
        close()
        if group.rank == 1:
            os._exit(17)

    group.close = fail_after_teardown
    execution._run_training(config, group=group, started=started, startup=startup, **args)


def main():
    from nn_window_ablation.cli import main as cli

    ddp_launch._rank_train = recovery_rank
    return cli()


if __name__ == "__main__":
    from tests.ddp_checks import main

    raise SystemExit(main())
