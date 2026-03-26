import os
import random

import numpy as np
import torch


def save_model(
    model, optim, scheduler, val_loss, val_cider, e, patience, best_cider, use_rl, args
):
    os.path.join(args.checkpoint_location, args.exp_name + "-last.pth")
    torch.save(
        {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
            "epoch": e,
            "val_loss": val_loss,
            "val_cider": val_cider,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "patience": patience,
            "best_cider": best_cider,
            "use_rl": use_rl,
        },
        f"{args.checkpoint_location}/{args.exp_name}_last.pth",
    )


def load_model(model, optim, scheduler, args, load_best=False):
    if load_best:
        fname = f"{args.checkpoint_location}/{args.exp_name}_best.pth"
    else:
        fname = f"{args.checkpoint_location}/{args.exp_name}_last.pth"

    data = torch.load(fname)
    torch.set_rng_state(data["torch_rng_state"])
    torch.cuda.set_rng_state(data["cuda_rng_state"])
    np.random.set_state(data["numpy_rng_state"])
    random.setstate(data["random_rng_state"])
    model.load_state_dict(data["state_dict"])

    if optim is None:  # or scheduler is None:
        return (
            model,
            None,
            None,
            data["epoch"] + 1,  # Start from the next epoch
            data["patience"],
            data["best_cider"],
            data["use_rl"],
        )

    optim.load_state_dict(data["optimizer"])
    scheduler.load_state_dict(data["scheduler"])

    return (
        model,
        optim,
        scheduler,
        data["epoch"] + 1,  # Start from the next epoch
        data["patience"],
        data["best_cider"],
        data["use_rl"],
    )
