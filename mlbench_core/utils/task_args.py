import argparse
import os


def task_main(main_func, uid="allreduce"):
    """Parses the task arguments and launches the main

    Args:
        main_func: Main function. Must have arguments `run_id`, `dataset_dir`, `ckpt_run_dir`, `output_dir`,
                    `rank`, `backend`, `hosts`, `validation_only`, `gpu`, `light_target`,
        uid: Task unique ID

    """
    dataset_dir, ckpt_run_dir, output_dir, args = _task_args(uid=uid)

    main_func(
        run_id=args.run_id,
        dataset_dir=dataset_dir,
        ckpt_run_dir=ckpt_run_dir,
        output_dir=output_dir,
        rank=args.rank,
        backend=args.backend,
        hosts=args.hosts,
        validation_only=args.validation_only,
        gpu=args.gpu,
        light_target=args.light,
    )


def _task_args(uid):
    """
    Parses the task arguments

    Args:
        uid (str): Task Unique ID

    Returns:
        str, str, str, dict: Dataset directory, checkpoint directory, output directory and arguments
    """
    parser = argparse.ArgumentParser(description="Process run parameters")
    parser.add_argument("--run_id", type=str, default="1", help="The id of the run")
    parser.add_argument(
        "--root-dataset",
        type=str,
        default="/datasets",
        help="Default root directory to dataset.",
    )
    parser.add_argument(
        "--root-checkpoint",
        type=str,
        default="/checkpoint",
        help="Default root directory to checkpoint.",
    )
    parser.add_argument(
        "--root-output",
        type=str,
        default="/output",
        help="Default root directory to output.",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        default=False,
        help="Only validate from checkpoints.",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Train with GPU"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        default=False,
        help="Train to light target metric goal",
    )
    parser.add_argument("--rank", type=int, default=1, help="The rank of the process")
    parser.add_argument(
        "--backend", type=str, default="mpi", help="PyTorch distributed backend"
    )
    parser.add_argument("--hosts", type=str, help="The list of hosts")

    args = parser.parse_args()

    dataset_dir = os.path.join(args.root_dataset, "torch", "wmt17")
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return dataset_dir, ckpt_run_dir, output_dir, args
