import os
import jax
import yaml
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse

# Print detected JAX devices
print("--- JAX Device Check ---")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("------------------------")

from vam.training import Trainer
from vam.config import get_config_from_cli


# This CLI trains a VAM on data specified in the command-line arguments.
# By default, a VAM will be trained using the default parameters.
# Set the -m flag to "task_opt" to train a task-optimized model and "binned_rt"
# to train a VAM on data from a specified RT quantile. See additional options below.

# For more control over the model architecture and training parameters,
# update the config struct returned by get_config_from_cli.

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data_dir",
    help="Directory with data used to train the model. If not specified, will read from config.yaml",
    default=None,
)
parser.add_argument(
    "-s",
    "--save_dir",
    help=(
        "Directory to save checkpoints and other info from model training. "
        "Note the info for this training run will be saved in "
        "the subfolder save_dir/expt_name. If not specified, will read from config.yaml"
    ),
    default=None,
)
parser.add_argument(
    "-e",
    "--expt_name",
    help=(
        "Name for this experiment/run as it will be logged by wandb. The info "
        "for this training run will be saved in save_dir/run_name. If not specified, will read from config.yaml"
    ),
    default=None,
)
parser.add_argument(
    "-p",
    "--project",
    nargs="?",
    type=str,
    help="Name of the project in wandb to log the run to, optional",
)
parser.add_argument(
    "-n",
    "--notes",
    nargs="?",
    type=str,
    help="Notes that will be attached to the run logged in wandb, optional",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="vam",
    nargs="?",
    type=str,
    help=(
        "Type of model to train. Set to 'vam' to train a VAM (default), "
        "'task_opt' to train a task-optimized model, and 'binned_rt' "
        "to train a VAM on data from a single RT quantile. If training "
        "a binned_rt model, the --rt_bin flag must be set, and optionally "
        "the n_rt_bins flag."
    ),
)
parser.add_argument(
    "--n_rt_bins",
    default=5,
    nargs="?",
    type=int,
    help=(
        "Number of RT quantiles to divide data into (default=5), "
        "only relevant for binned_rt model training."
    ),
)
parser.add_argument(
    "--rt_bin",
    nargs="?",
    type=int,
    help=(
        "Which RT quantile to train the data on, should be an integer "
        "between 0 and n_rt_bins-1 (inclusive), only relevant for "
        "binned_rt model training."
    ),
)

def load_config_defaults(args):
    """Load default values from config.yaml for any unspecified arguments."""
    # Check if any arguments need to be loaded from config
    needs_config = args.data_dir is None or args.save_dir is None or args.expt_name is None
    
    if needs_config:
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
            with open("config.yaml", "r") as f:
                config_data = yaml.safe_load(f)
                vam_config = config_data["vam"]
                
                # Load missing arguments from config
                if args.data_dir is None:
                    args.data_dir = vam_config["data_dir"]
                    print(f"Using data_dir from config.yaml: {args.data_dir}")
                
                if args.save_dir is None:
                    args.save_dir = vam_config["save_dir"]
                    print(f"Using save_dir from config.yaml: {args.save_dir}")
                
                if args.expt_name is None:
                    args.expt_name = vam_config["expt_name"]
                    print(f"Using expt_name from config.yaml: {args.expt_name}")
                    
        except FileNotFoundError:
            print("Error: config.yaml not found and required arguments not provided via command line")
            exit(1)
        except KeyError as e:
            print(f"Error: Could not find required key in config.yaml structure: {e}")
            exit(1)
        except Exception as e:
            print(f"Error reading config.yaml: {e}")
            exit(1)

args = parser.parse_args()
load_config_defaults(args)

config = get_config_from_cli(args)

print(config)
trainer = Trainer(config, args.save_dir, args.data_dir)
trainer.train()



# Test -------------------------------------------------
"""
Example usage:        
# Using all defaults from config.yaml
python train_model.py

# Specifying some arguments explicitly (others from config.yaml)
python train_model.py --save_dir custom_output/ --expt_name my_experiment

# Specifying all arguments explicitly
python train_model.py \
    --data_dir /path/to/data \
    --save_dir /path/to/output \
    --expt_name experiment_namw

# Using short flags
python train_model.py -d /path/to/data -s /path/to/output -e experiment_name

# Example:
python train_model.py \
    --data_dir /home/fs0/jdf650/scratch/VAMSalience/model_inputs_augmented/brightness \
    --save_dir out/ \
    --expt_name test
"""