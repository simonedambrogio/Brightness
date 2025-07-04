import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn # Import Flax Linen
from tqdm import tqdm # Import tqdm here for the main loop
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import model definition and config loading utils
from vam.models import CNN, ModelConfig
from vam.load_trained_model import load_trained_model

def load_image_batch(data_dir, start_idx, batch_size, n_total_images, img_shape=(128, 128, 3), dtype=np.float32):
    """Loads a specific batch of images starting from start_idx."""
    # Determine the end index, ensuring it doesn't exceed total images
    end_idx = min(start_idx + batch_size, n_total_images)
    actual_batch_size = end_idx - start_idx
    
    if actual_batch_size <= 0:
        print("Warning: No images to load for this batch range.")
        return None
        
    # print(f"\n--- Loading images {start_idx} to {end_idx-1} (batch size {actual_batch_size}) from {data_dir} ---")
    stimuli_dir = os.path.join(data_dir, 'processed_stimuli')
    loaded_images = []
    expected_shape = img_shape
    
    try:
        # Iterate through the specific indices for this batch
        for i in range(start_idx, end_idx):
            img_path = os.path.join(stimuli_dir, f'img{i}.npy')
            img = np.load(img_path, allow_pickle=False)
            if img.shape != expected_shape:
                 print(f"Warning: Image {i} shape {img.shape} differs from expected {expected_shape}. Skipping? (Modify if needed)")
                 continue
            loaded_images.append(img.astype(dtype)) # Ensure correct dtype
            
        if not loaded_images:
            print("Error: No images loaded successfully for this batch.")
            return None
            
        image_batch = np.stack(loaded_images, axis=0)
        # print(f"Loaded image batch with shape: {image_batch.shape}")
        return image_batch
    except FileNotFoundError as e:
        print(f"Error loading image: {e}. Ensure path is correct and files exist up to index {n_total_images-1}.")
        return None
    except Exception as e:
        print(f"Error during image loading: {e}")
        return None

def extract_dense1_input(model_config, params, image_batch):
    """Extracts the activations that are input to the final Dense layer (Dense_1)."""
    # print("\n--- Extracting Dense_1 input activations ---")
    
    # Define a temporary model capturing layers *before* the final Dense layer
    class CNN_before_last(nn.Module):
        config: ModelConfig
        param_dtype: jnp.dtype

        @nn.compact
        def __call__(self, x, training: bool):
            # Replicate layers from original CNN up to before Dense_1
            # Using names ensures parameter matching if original used named layers
            for i, n in enumerate(self.config.conv_n_features):
                 x = nn.Conv(features=n, kernel_size=(3, 3), padding="SAME", dtype=self.param_dtype, name=f"Conv_{i}")(x)
                 x = nn.relu(x)
                 x = nn.GroupNorm(num_groups=n, name=f"GroupNorm_{i}")(x)
                 x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                 
            x = x.reshape((x.shape[0], -1))
            
            for i, n in enumerate(self.config.fc_n_units):
                 x = nn.Dense(features=n, dtype=self.param_dtype, name=f"Dense_{i}")(x)
                 x = nn.relu(x)
                #  x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training, name=f"Dropout_{i}")(x)
                 
            # Return the activations *before* the final Dense layer is applied
            return x 

    try:
        temp_model = CNN_before_last(model_config, model_config.param_dtype)
        
        # Apply the temporary model using the loaded parameters for the CNN part
        # Note: We assume params here are JUST the CNN params ('get_drifts')
        activations = temp_model.apply({'params': params}, image_batch, training=False)
        
        # print(f"Extracted activations shape: {activations.shape}")
        return np.array(activations) # Return as numpy array
    except Exception as e:
        print(f"Error during activation extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_activations(activations, output_dir, checkpoint, batch_idx, layer_name="Dense_1_input"):
    """Saves the extracted activations for a specific batch."""
    # print(f"\n--- Saving activations for {layer_name}, Batch {batch_idx} ---")
    
    checkpoint_str = f"ckpt{checkpoint}" if checkpoint is not None else "latest"
    batch_str = f"batch{batch_idx:04d}" # Padded batch index
        
    filename = f"activations_{layer_name}_{checkpoint_str}_{batch_str}.npy"
    output_path = os.path.join(output_dir, filename)
    
    try:
        np.save(output_path, activations)
        # print(f"Successfully saved batch activations to {output_path}")
    except Exception as e:
        print(f"Error saving batch activations to {output_path}: {e}")

def instantiate_and_print_cnn(config):
    """Instantiates CNN model, creates dummy input, prints architecture."""
    try:
        # Get the model config dict (handle ConfigDict if necessary)
        if hasattr(config.model, 'to_dict'):
             model_config_dict = config.model.to_dict()
        else:
             model_config_dict = config.model.copy() # Make a copy if it's already a dict
             
        # Map dtype string/type to jax dtype
        loaded_dtype = model_config_dict.get('param_dtype') # Get original dtype
        jax_dtype = jnp.float32 # Default
        if isinstance(loaded_dtype, jnp.dtype):
            jax_dtype = loaded_dtype
        elif isinstance(loaded_dtype, (np.dtype, type)):
            dtype_map = {np.float32: jnp.float32, np.float64: jnp.float64, np.int32: jnp.int32, np.int64: jnp.int64}
            jax_dtype = dtype_map.get(loaded_dtype, jnp.float32)
            if jax_dtype == jnp.float32 and loaded_dtype not in dtype_map:
                 print(f"Warning: Unsupported numpy dtype {loaded_dtype}. Defaulting to jnp.float32.")
        elif isinstance(loaded_dtype, str):
             try:
                  jax_dtype = getattr(jnp, loaded_dtype)
             except AttributeError:
                  print(f"Warning: Could not map dtype string '{loaded_dtype}'. Defaulting to jnp.float32.")
        elif loaded_dtype is not None:
             print(f"Warning: Unrecognized dtype format {type(loaded_dtype)}. Defaulting to jnp.float32.")
        else:
             print("Warning: param_dtype not found in config. Defaulting to jnp.float32.")
             
        # Update the dtype in the dictionary *before* unpacking
        model_config_dict['param_dtype'] = jax_dtype

        # Instantiate ModelConfig using the updated dictionary
        model_config = ModelConfig(**model_config_dict)
        
        cnn_model = CNN(model_config, model_config.param_dtype)
        print("CNN model instantiated.")
    except AttributeError as e:
        print(f"Error: Missing attribute in config needed for ModelConfig: {e}")
        print("Config model dict:", config.model)
        return None, None, None, None # Return None on error
    except Exception as e:
        print(f"Error instantiating CNN model: {e}")
        return None, None, None, None # Return None on error

    # Create a dummy input (batch_size=1, height, width, channels)
    img_h = getattr(config.data, 'img_h', 128)
    img_w = getattr(config.data, 'img_w', 128)
    dummy_input = jnp.ones((1, img_h, img_w, 3), dtype=jax_dtype)
    print(f"Using dummy input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")

    # Generate a PRNG key
    key = jax.random.PRNGKey(0)

    # Print the architecture table
    print_jax_table(cnn_model, dummy_input, key)
    
    return cnn_model, dummy_input, key, model_config # Return model, input, key, model_config

def print_jax_table(model, dummy_input, key):
    """Prints a summary table of the JAX/Flax model architecture."""
    print("\n--- Model Architecture Summary ---")
    try:
        # depth=1 limits nesting in table, remove or increase for more detail
        summary_table = model.tabulate(
            key,
            dummy_input, 
            training=False, # Use training=False for typical architecture view
            # Optional args for more detail:
            # compute_flops=True, 
            # compute_vjp_flops=True,
            # depth=None
        )
        print(summary_table)
    except Exception as e:
        print(f"Error generating model summary: {e}")
        import traceback
        traceback.print_exc()

def loading_trained_model(experiment_dir, checkpoint=None):
    config, restored_state = load_trained_model(
        experiment_dir, 
        specific_checkpoint=checkpoint
    )
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    # Check if state loading failed when a specific checkpoint was requested
    if restored_state is None and args.checkpoint is not None:
        print(f"Error: Configuration loaded, but failed to load state for checkpoint {args.checkpoint}. Cannot save weights.")
        return # Exit if state is needed but failed to load
    elif restored_state is None:
         print(f"Warning: Failed to load latest model state. Only printing architecture.")

    print("Configuration loaded successfully.")
    
    return config, restored_state

def main(args):
    print(f"\n--- Loading configuration and state for experiment: {args.experiment_dir} ---")
    config, restored_state = loading_trained_model(
        args.experiment_dir, 
        checkpoint=args.checkpoint)
    
    # Instantiate model and print architecture
    cnn_model, dummy_input, key, model_config = instantiate_and_print_cnn(config)

    if cnn_model is None:
        print("Exiting due to model instantiation error.")
        return
        
    if restored_state is None:
        print("\nCannot proceed without loaded model state. Exiting.")
        return
        
    # Extract the CNN-specific parameters ('get_drifts') from the full state
    if 'get_drifts' not in restored_state.params:
         print("Error: 'get_drifts' parameters not found in loaded state.")
         return
    cnn_params = restored_state.params['get_drifts']

    # --- Process Images in Batches --- #
    batch_size = 256 
    n_total = args.n_total_images
    print(f"\n--- Starting batch processing for {n_total} images (batch size: {batch_size}) ---")

    all_activations = [] # List to store activations if needed later, use with caution for memory

    # Wrap the range iterator with tqdm for batch progress
    batch_iterator = range(0, n_total, batch_size)
    for i in tqdm(batch_iterator, desc="Processing Batches"):
        batch_idx = i // batch_size
        # Optional: Remove the print statement below if tqdm bar is sufficient
        # print(f"\n>>> Processing Batch {batch_idx} (Indices {i} to {min(i+batch_size, n_total)-1}) <<<")
        
        # --- Load image batch --- # 
        image_batch = load_image_batch(
            args.data_dir, 
            start_idx=i, 
            batch_size=batch_size, 
            n_total_images=n_total,
            img_shape=dummy_input.shape[1:], 
            dtype=np.dtype(dummy_input.dtype.name)
        )
        
        if image_batch is None:
            # Optional: print warning or log error if needed
            continue # Skip to next batch
            
        # --- Extract Activations --- # 
        dense1_input_activations = extract_dense1_input(model_config, cnn_params, image_batch)

        # --- Save Activations --- # 
        if dense1_input_activations is not None:
            save_activations(
                dense1_input_activations, 
                args.output_dir, 
                args.checkpoint, 
                batch_idx=batch_idx, # Pass batch index
                layer_name="Dense_1_input"
            )
        else:
            print(f"Skipping saving activations for batch {batch_idx} due to extraction error.")
            
    print(f"\n--- Batch processing finished for {n_total} images --- ")

if __name__ == "__main__":
    """Example usage:
        python save_network_weights.py analysis_env/brightness_augmented
        
        python save_network_weights.py \
            training_outputs/attentional_augmented_brightness_env/attentional_augmented_brightness \
            --data_dir model_inputs_augmented/brightness \
            --output_dir analysis_output/attentional/brightness/weights \
            --checkpoint 69 \
            --n_total_images 35000
    """
    parser = argparse.ArgumentParser(description="Print CNN architecture summary from a trained VAM experiment.")
    parser.add_argument("experiment_dir", type=str,
                        help="Path to the specific experiment directory containing config (e.g., analysis_env/both).")
    # Add optional argument for specific checkpoint step
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Specific checkpoint step to load configuration for. Uses latest if not specified.")
    # Add arguments for data loading and output saving
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing input data (behavioral files, processed_stimuli folder).")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                        help="Directory to save the extracted weights.")
    parser.add_argument("--n_total_images", type=int, default=35000, # New argument
                        help="Total number of images to process.")

    args = parser.parse_args()

    # --- Create output directory --- #
    # Use a subdirectory for weights to keep things organized
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)
        
    main(args) # Pass the specific dir to main

