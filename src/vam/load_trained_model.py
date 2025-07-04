"""Example usage:
python load_trained_model.py analysis_env/cuda_experiment
"""

import argparse
import os
import jax
import jax.numpy as jnp
# import orbax.checkpoint as ocp # No longer using Orbax directly for restore
import ml_collections
import yaml
import flax.core
import optax
import numpy as np # Need numpy for initial dtype parsing if necessary
from flax.training import checkpoints # Use Flax checkpoints for restore

# Import necessary components from your project
from vam.models import ModelConfig, VAM, CNN
# Import the specific functions needed, including the modified optimizer helpers
from vam.training import TrainState, flattened_traversal, vam_label_fn, _get_cnn_opt, _get_vi_opt 
from vam.config import get_default_config # To help structure the loaded config

# ==================================
# Step 1: Load Configuration
# ==================================
def load_configuration(experiment_dir):
    """Loads the config.yaml saved by W&B."""
    # W&B dir is usually parallel to the experiment dir (e.g., analysis_env/wandb)
    experiment_dir_cleaned = os.path.normpath(experiment_dir)
    parent_dir = os.path.dirname(os.path.abspath(experiment_dir_cleaned))
    if not parent_dir or parent_dir == os.path.abspath(experiment_dir_cleaned):
        parent_dir = "."
    wandb_dir = os.path.join(parent_dir, 'wandb')

    config_path_rel = os.path.join('latest-run', 'files', 'config.yaml')
    config_path_abs = os.path.join(wandb_dir, config_path_rel)

    if not os.path.exists(config_path_abs):
        if not os.path.isdir(wandb_dir):
            print(f"Error: W&B directory not found at {wandb_dir}")
            return None
        run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith('run-') and os.path.isdir(os.path.join(wandb_dir, d))]
        if not run_dirs:
            print(f"Error: Could not find any W&B run directory in {wandb_dir}")
            return None
        latest_run_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(wandb_dir, d)))
        print(f"Warning: 'latest-run' link not found or invalid in {wandb_dir}. Using most recent run dir: {latest_run_dir}")
        config_path_abs = os.path.join(wandb_dir, latest_run_dir, 'files', 'config.yaml')

    if not os.path.exists(config_path_abs):
        print(f"Error: config.yaml not found at {config_path_abs} or in latest run dir.")
        return None

    print(f"Loading config from: {config_path_abs}")
    with open(config_path_abs, 'r') as f:
        wandb_config_raw = yaml.safe_load(f)

    config_dict = {}
    for key, item in wandb_config_raw.items():
        if key == '_wandb':
            continue
        if isinstance(item, dict) and 'value' in item:
            if isinstance(item['value'], dict):
                config_dict[key] = ml_collections.ConfigDict(item['value'])
                for sub_key, sub_value in item['value'].items():
                    if isinstance(sub_value, list):
                        config_dict[key][sub_key] = list(sub_value)
            else:
                config_dict[key] = item['value']

    config = ml_collections.ConfigDict(config_dict)
    for key, value in config.items():
        if isinstance(value, dict) and not isinstance(value, ml_collections.ConfigDict):
            config[key] = ml_collections.ConfigDict(value)
    
    print("Successfully loaded configuration.")
    return config

# ===========================================
# Step 2: Initialize Model Structures
# ===========================================
def _get_cnn_opt(config):
    assert config.optimizer.cnn_opt in ["sgd", "adam", "adamw"], "Invalid CNN optimizer"
    lr = config.optimizer.cnn_lr
    if config.optimizer.cnn_opt == "sgd": return optax.sgd(lr, config.optimizer.cnn_momentum)
    if config.optimizer.cnn_opt == "adam": return optax.adam(lr)
    if config.optimizer.cnn_opt == "adamw": return optax.adamw(lr)

def _get_vi_opt(config):
    assert config.optimizer.vi_opt in ["adam"], "Invalid VI optimizer"
    if config.optimizer.vi_opt == "adam": return optax.adam(config.optimizer.vi_lr)

def initialize_model_structures(config, root_key):
    """Initializes the model structure based on config to get shapes."""
    print("Initializing model structure for type:", config.model.model_type)
    
    # Handle potential 'numpy.' prefix in dtype from config
    param_dtype_str = config.model.param_dtype
    if param_dtype_str.startswith('numpy.'):
        param_dtype_str = param_dtype_str.replace('numpy.', '')
        print(f"  Adjusted param_dtype from config to: {param_dtype_str}")

    param_dtype_jax = getattr(jnp, param_dtype_str)

    # Create dummy inputs with correct shapes and dtypes
    # Use batch size 1 for initialization
    # Construct image_shape from img_h and img_w in config.data
    if not isinstance(config.get('data'), ml_collections.ConfigDict):
        raise TypeError("Configuration 'data' section is missing or not a ConfigDict.")
    img_h = config.data.get('img_h')
    img_w = config.data.get('img_w')
    if img_h is None or img_w is None:
        raise KeyError("Configuration is missing 'img_h' or 'img_w' under the 'data' section.")
    # Assuming 3 channels (H, W, C)
    image_shape = [img_h, img_w, 3]
    print(f"  Constructed image shape: {image_shape}")
    dummy_img_shape = tuple([1] + image_shape) # (1, H, W, C)

    dummy_rts_shape = (1,) # (1,)
    dummy_choices_shape = (1,) # (1,)
    dummy_gaze_shape = (1, config.model.n_acc) # (1, n_acc) - Added for attentional model

    dummy_imgs = jnp.zeros(dummy_img_shape, dtype=param_dtype_jax)
    dummy_rts = jnp.zeros(dummy_rts_shape, dtype=jnp.float32) # Typically float
    dummy_choices = jnp.zeros(dummy_choices_shape, dtype=jnp.int32) # Typically int
    dummy_gaze = jnp.zeros(dummy_gaze_shape, dtype=jnp.float32) # Typically float - Added

    # Split key for initialization
    k1, k2 = jax.random.split(root_key)

    if config.model.model_type == 'vam':
        model = VAM(ModelConfig(**config.model), param_dtype=param_dtype_jax)
        # Use eval_shape to get param structure without computation
        params_struct = jax.eval_shape(
            # Provide all required args to model.init (which calls __call__ indirectly)
            lambda k1, k2: model.init(k1, dummy_imgs, dummy_rts, dummy_choices, dummy_gaze, k2, training=False)['params'],
            k1, k2 # Pass example args matching the lambda signature
        )
        # Initialize optimizer state shape
        # Reconstruct the optimizer using the logic from create_vam_train_state
        label_fn = flattened_traversal(vam_label_fn)
        cnn_opt = _get_cnn_opt(config) # Pass the loaded config
        vi_opt = _get_vi_opt(config)   # Pass the loaded config
        optimizer = optax.chain(
            optax.clip(config.optimizer.clip_val),
            optax.multi_transform(
                {"cnn": cnn_opt, "vi": vi_opt},
                label_fn,
            ),
        )
        # Unfreeze the parameter structure before passing to optimizer init
        mutable_params_struct = flax.core.unfreeze(params_struct)
        opt_state_struct = jax.eval_shape(optimizer.init, mutable_params_struct)

    # elif config.model.model_type == 'other_model':
    #     # Handle other model types if needed
    #     pass
    else:
        raise ValueError(f"Unknown model type specified in config: {config.model.model_type}")

    print("Successfully determined model and optimizer state structures.")
    # Return the individual structures and instances needed
    return params_struct, opt_state_struct, model, optimizer

# ===================================
# Step 3: Create Template State
# ===================================
def create_template_state(model, params_struct, optimizer, opt_state_struct, analysis_key):
    """Creates the template TrainState structure and the target dictionary for restoration."""
    # Create a dummy TrainState instance using the derived structures
    # Note: apply_fn comes from the model instance
    # Note: step needs a dummy value (0 is fine)
    # Note: key needs a dummy value (analysis_key)
    template_state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params_struct, # Use param structure
        tx=optimizer,         # Use optimizer instance
        opt_state=opt_state_struct, # Use optimizer state structure
        key=analysis_key       # Use provided key
    )
    template_target = {'model': template_state} # Match the structure saved during training
    print("Template TrainState structure created for restoration target.")
    return template_target

# ==========================================
# Step 4: Find and Restore Checkpoint
# ==========================================
def restore_latest_checkpoint(experiment_dir, template_target, specific_step=None):
    """Finds the latest (or specified) checkpoint and restores it using Flax."""
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        print(f"Error: Checkpoint directory not found at {ckpt_dir}")
        return None

    print(f"Looking for checkpoints in: {ckpt_dir}")

    target_step = specific_step

    if target_step is None:
        # Find the latest step if none specified
        latest_step = None
        try:
            steps = []
            for item in os.listdir(ckpt_dir):
                item_path = os.path.join(ckpt_dir, item)
                if os.path.isdir(item_path) and item.startswith('checkpoint_'):
                    try:
                        step_num = int(item.split('_')[-1])
                        steps.append(step_num)
                    except (ValueError, IndexError):
                        continue
            if steps:
                latest_step = max(steps)
        except OSError as e:
            print(f"Error listing checkpoint directory {ckpt_dir}: {e}")
            return None

        if latest_step is None:
            print(f"Error: No checkpoints found matching 'checkpoint_*' pattern in {ckpt_dir}")
            return None
        target_step = latest_step
        print(f"Found latest checkpoint step: {target_step}")
    else:
        print(f"Attempting to load specified checkpoint step: {target_step}")
        # Verify the specified checkpoint directory exists
        expected_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{target_step}")
        if not os.path.isdir(expected_ckpt_path):
            print(f"Error: Specified checkpoint directory not found: {expected_ckpt_path}")
            return None

    print(f"Restoring checkpoint from step {target_step} using Flax...")

    restored_target = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=template_target,
        step=target_step # Use the target step
    )

    if restored_target is None:
        print(f"Error: Failed to restore checkpoint for step {target_step} from {ckpt_dir}")
        return None
    
    restored_state = restored_target['model']
    print(f"Successfully restored state from step {target_step}")
    # Return only the state, config is available in the caller (main)
    return restored_state

# ==============================================
# Step 5: Orchestrator Function
# ==============================================
def load_trained_model(experiment_dir, specific_checkpoint=None):
    """Loads configuration and restores the latest or a specific trained model state.

    Args:
        experiment_dir: Path to the experiment directory.
        specific_checkpoint (int, optional): Specific checkpoint step to load. 
                                             Loads latest if None. Defaults to None.

    Returns:
        A tuple (config, restored_state) or (None, None) if loading fails.
    """
    print(f"\n--- Loading Trained Model for Experiment: {experiment_dir} ---")

    # 1. Load Configuration
    config = load_configuration(experiment_dir)
    if config is None:
        print("Failed during configuration loading.")
        return None, None

    # 2. Initialize Model Structures
    # Use a fixed key for analysis initialization
    root_key = jax.random.PRNGKey(config.training.seed if hasattr(config.training, 'seed') else 42)
    init_results = initialize_model_structures(config, root_key)
    if init_results is None: # Check if model initialization failed (adjusted check)
        print("Failed during model structure initialization.")
        return None, None
    # Unpack the returned structures and instances
    params_struct, opt_state_struct, model, optimizer = init_results

    # 3. Create Template State
    # Use a different key for the state template if needed
    analysis_key = jax.random.fold_in(root_key, 1) # Example: Fold in for a different key
    template_target = create_template_state(model, params_struct, optimizer, opt_state_struct, analysis_key)

    # 4. Restore Checkpoint (pass specific_checkpoint)
    restored_target = restore_latest_checkpoint(experiment_dir, template_target, specific_step=specific_checkpoint)
    if restored_target is None:
        print("Failed during checkpoint restoration.")
        return None, None

    print("\n--- Model Loading Complete ---")
    return config, restored_target

def _run_example_usage():
    """Helper function for the example usage block."""
    parser = argparse.ArgumentParser(description="Load a trained VAM model.")
    parser.add_argument("experiment_dir", type=str,
                        help="Path to the specific experiment directory (e.g., analysis_env/cuda_experiment)")
    # Add the optional checkpoint argument to the example usage as well
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Specific checkpoint step to load. Loads latest if not specified.")
    args = parser.parse_args()

    # Pass the argument to the main loading function
    loaded_config, loaded_state = load_trained_model(args.experiment_dir, specific_checkpoint=args.checkpoint)

    if loaded_state:
        print("\nSuccessfully loaded model.")
        print(f"  Config Model Type: {loaded_config.model.model_type}")
        print(f"  Restored Step: {loaded_state.step}")
        # You can add further checks or print details about loaded params here
        # Example: Check structure
        # print(jax.tree_map(lambda x: (x.shape, x.dtype) if hasattr(x, 'shape') else x, loaded_state.params))
    else:
        print("\nModel loading failed.")

# Example usage block (optional, can be commented out or removed)
if __name__ == "__main__":
    _run_example_usage()
