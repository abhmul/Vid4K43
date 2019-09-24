import argparse
import json
import os

from kaggleutils import safe_create_file

EXPERIMENT_DIR = "../experiment_specs"


def args_to_dict(args):
    return {k: v for k, v in vars(args).items() if v is not None}


def read_experiment(args):
    experiment_path = os.path.join(EXPERIMENT_DIR, f"{args.experiment}.json")
    if args.load == "":
        print(f"Loading experiment from {experiment_path}\n")
        with open(safe_create_file(experiment_path), "r"):
            d = json.load(experiment_file)
        for arg, value in d.items():
            vars(args)[arg] = value

    else:
        print(f"Saving experiment to {experiment_path}\n")
        d = args_to_dict(args)
        with open(safe_create_file(experiment_path), "w") as experiment_file:
            json.dump(d, experiment_file)

    print(f"Running experiment {args.experiment} with args {d}")
    return args


def ExperimentParser(*args, **kwargs):
    parser = argparse.ArgumentParser(*args, **kwargs)
    parser.add_argument("-e", "--experiment", help="Name of experiment")
    parser.add_argument(
        "--load", action="store_true", help="Load an experiment by name"
    )
    return parser
