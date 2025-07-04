import os
import pickle
import pdb

import jax.numpy as jnp
from jax import random, vmap
import numpy as np
from flax import traverse_util
import scipy
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import jax


def plot_batch_imgs(x, y, figsize=(12, 8), num_rows=5, num_columns=3, title=None):
    """Plots a batch of images x with labels y."""

    if len(x) != len(y):
        raise ValueError("Number of images and number of labels don't match!")

    _, ax = plt.subplots(num_rows, num_columns, figsize=figsize)

    for i in range(num_rows * num_columns):
        try:
            img = x[i]
            label = str(y[i])
            ax[i // num_columns, i % num_columns].imshow(img)
            ax[i // num_columns, i % num_columns].set_title(label)
            ax[i // num_columns, i % num_columns].axis("off")
        except:
            pass

    plt.tight_layout()
    plt.show()


def constant_init(value, dtype="float32"):
    """Initialize flax linen module parameters to value (a constant)."""

    def _init(key, shape, dtype=dtype):
        return value * jnp.ones(shape, dtype)

    return _init


def reparam_fr(key, mu, L):
    """Reparameterization trick for a full-rank Gaussian parameterized by a
    lower triangular covariance matrix L and mean vector mu."""
    sample = random.multivariate_normal(key, jnp.zeros(len(mu)), jnp.eye(len(mu)))
    return mu + L @ sample


batch_reparam_fr = vmap(reparam_fr, in_axes=[0, None, None])


def vec_to_lowertri(v, D):
    """Transform vector v to a DxD lower triangular matrix."""
    tril_idx = jnp.tril_indices(D)
    L = jnp.zeros((D, D))
    return L.at[tril_idx].set(v)


def lowertri_to_vec(L, D):
    """Transform DxD lower triangular matrix L to a vector."""
    return L[jnp.tril_indices(D)]


def lba_jac_adj(A_raw, c_raw, t0_raw, gamma_raw):
    """Jacobian adjustment for the VAM.
    Note: This function receives raw parameters (A_raw, c_raw, t0_raw, gamma_raw) from models.py,
    despite the signature. The calculation A_raw+c_raw+t0_raw+gamma_raw yields the correct
    log Jacobian determinant (A_raw+c_raw+t0_raw+gamma_raw) for the exp() transformations.
    """
    return A_raw + c_raw + t0_raw + gamma_raw
    # return jnp.log(jnp.abs(A * c * t0 * gamma))


def vam_label_fn(path, x):
    if path[0] == "get_drifts":
        return "cnn"
    else:
        return "vi"


def flattened_traversal(fn):
    def mask(tree):
        flat = traverse_util.flatten_dict(tree)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def get_vam_lba_params(state, save_path=None):
    c_raw = state.params["get_elbo"]["c"]
    a_raw = state.params["get_elbo"]["a"]
    t0_raw = state.params["get_elbo"]["t0"]
    gamma_raw = state.params["get_elbo"]["gamma"]
    
    c = jnp.exp(c_raw)
    a = jnp.exp(a_raw)
    t0 = jnp.exp(t0_raw)
    gamma = jax.nn.sigmoid(gamma_raw)
    b = c + a
    
    params = {"a": a, "b": b, "t0": t0, "gamma": gamma}

    if save_path is not None:
        params_np = {k: np.array(v) for k, v in params.items()}
        with open(save_path, "wb") as f:
            pickle.dump(params_np, f)

    return params


def get_wandb_info(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    user_list, expt_id_list = [], []
    for rr in runs:
        try:
            user_list.append(rr.name[4:])
            expt_id_list.append(rr.id)
        except:
            continue

    info_df = pd.DataFrame({"user_id": user_list, "expt_id": expt_id_list})

    return info_df
