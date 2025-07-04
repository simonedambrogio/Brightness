import os
import pdb
import time

import jax
from jax import random
import jax.numpy as jnp
import optax
import orbax.checkpoint as orbax
from flax.training import train_state, checkpoints
import flax.core
import flaxmodels as fm
import h5py
import wandb
import ml_collections

from .metrics import (
    TrainMetrics,
    VAMMetrics,
    TaskOptMetrics,
    get_vam_user_metrics,
    log_metrics,
)
from .utils import (
    flattened_traversal,
    vam_label_fn,
    get_vam_lba_params,
    plot_batch_imgs,
)
from .task_data import TaskData
from .lba import generate_vam_rts
from .models import ModelConfig, VAM, CNN


class TrainState(train_state.TrainState):
    key: random.KeyArray


@jax.jit
def vam_train_step(state: train_state.TrainState, batch, mc_key, dropout_key):
    imgs, _salience, rts, choices, g = batch[0], batch[1], batch[2], batch[3], batch[4]
    dropout_train_key = random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        elbo, _ = state.apply_fn(
            {"params": params},
            imgs,
            rts,
            choices,
            g,
            mc_key,
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        return -elbo

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = TrainMetrics.single_from_model_output(loss=loss)
    return state, metrics


@jax.jit
def vam_eval_step(state: train_state.TrainState, batch, root_key):
    root_key, mc_key = random.split(root_key)
    imgs, _salience, rts, choices, g = (
        batch[0],
        batch[1],
        batch[2],
        batch[3],
        batch[4],
    )
    elbo, drifts = state.apply_fn(
        {"params": state.params}, imgs, rts, choices, g, mc_key, training=False
    )
    lba_params = get_vam_lba_params(state)

    root_key, mc_key = random.split(root_key)
    n_acc = drifts.shape[-1]
    sim_data = generate_vam_rts(lba_params, drifts, n_acc, mc_key)
    metrics = VAMMetrics.single_from_model_output(
        loss=-elbo,
        rts=sim_data["rts"],
        responses=sim_data["response"],
        valid_idx=sim_data["valid_idx"],
        drifts=drifts,
    )
    return state, root_key, metrics, lba_params


@jax.jit
def task_opt_train_step(state: train_state.TrainState, batch, dropout_key):
    imgs, targets = batch[0], batch[3]
    dropout_train_key = random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            imgs,
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=targets
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = TaskOptMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=targets
    )
    return state, metrics


@jax.jit
def task_opt_eval_step(state: train_state.TrainState, batch):
    imgs, targets = batch[0], batch[3]

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            imgs,
            training=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=targets
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = TaskOptMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=targets
    )
    return state, metrics


def _get_cnn_opt(config):
    assert config.optimizer.cnn_opt in [
        "sgd",
        "adam",
        "adamw",
    ], "CNN optimizer must be 'sgd', 'adam', or 'adamw'"
    lr = config.optimizer.cnn_lr
    if config.optimizer.cnn_opt == "sgd":
        opt = optax.sgd(lr, config.optimizer.cnn_momentum)
    elif config.optimizer.cnn_opt == "adam":
        opt = optax.adam(lr)
    elif config.optimizer.cnn_opt == "adamw":
        opt = optax.adamw(lr)
    return opt


def _get_vi_opt(config):
    assert config.optimizer.vi_opt in ["adam"], "VI optimizer must be 'adam'"
    if config.optimizer.vi_opt == "adam":
        opt = optax.adam(config.optimizer.vi_lr)
    return opt


class Trainer:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
        save_dir,
        data_dir,
        data_save_dir=None,
        reload_epoch=None,
        wandb_id=None,
        split_dir=None,
    ):
        self.config = config
        self.model_type = config.model.model_type  # either vam or task_opt
        self.root_key = random.PRNGKey(seed=config.training.seed)
        self.expt_ckpt_dir = os.path.join(save_dir, config.expt_name, "checkpoints")
        self.checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())

        # Set up data loader
        self.root_key, data_split_key = random.split(self.root_key)
        self.task_data = TaskData(
            data_dir,
            save_dir,
            data_split_key,
            config,
            data_save_dir=data_save_dir,
            split_dir=split_dir,
        )

        # Set up models
        self.root_key, self.dropout_key = random.split(self.root_key)
        if self.model_type == "task_opt":
            model = CNN(ModelConfig(**config.model), self.config.model.param_dtype)
            self.state = self.create_task_opt_train_state(model)
        elif self.model_type in ["vam", "binned_rt"]:
            model = VAM(ModelConfig(**config.model), self.config.model.param_dtype)
            self.state = self.create_vam_train_state(model)

        # Initialize CNN if testing
        if config.model.test_only:
            self.cnn = CNN(ModelConfig(**config.model), self.config.model.param_dtype)
            self.root_key, params_key = random.split(self.root_key, num=2)
            img_h, img_w = config.data.img_h, config.data.img_w
            imgs = jnp.ones([2, img_h, img_w, 3])
            _ = self.cnn.init(params_key, imgs, training=False)["params"]

        # Check for epoch to load checkpoint from
        if reload_epoch is not None:
            target = {"model": self.state}
            state_restored = checkpoints.restore_checkpoint(
                self.expt_ckpt_dir, target=target, step=reload_epoch
            )
            self.state = state_restored["model"]
            self.start_epoch = reload_epoch + 1
            if not self.config.model.test_only:
                wandb_resume = "must"
        else:
            wandb_resume = None
            self.start_epoch = 0

        # Start experiment logger
        if not self.config.model.test_only:
            wandb.init(
                dir=save_dir,
                project=config.project,
                config=config.to_dict(),
                notes=config.notes,
                tags=config.tags,
                name=config.expt_name,
                resume=wandb_resume,
                id=wandb_id,
            )

    def _transfer_vgg_weights(self, params, imgs):
        # Initialize pretrained VGG and transfer parameters to CNN, VGG from:
        # https://github.com/matthias-wright/flaxmodels/tree/main/flaxmodels/vgg

        weights_path = "vgg16_weights.h5"

        # Initialize VGG model structure without attempting download
        vgg_model = fm.VGG16(
            output="logits",
            pretrained=None, # Initialize without weights first
            include_head=False,
            normalize=False,
        )
        vgg_rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        # Initialize parameters (will be random or default initially)
        vgg_params = vgg_model.init(vgg_rngs, imgs)['params']

        # Manually load weights from local file if it exists
        if os.path.exists(weights_path):
            print(f"Loading VGG weights manually from: {weights_path}")
            vgg_params = vgg_params.unfreeze()
            with h5py.File(weights_path, 'r') as f:
                # Load weights layer by layer, assuming h5 structure matches flax param structure
                for layer_name, layer_params in vgg_params.items():
                    if layer_name in f:
                        for param_name, param_value in layer_params.items():
                            # Map model's 'kernel' to H5's 'weight'
                            h5_param_name = 'weight' if param_name == 'kernel' else param_name

                            if h5_param_name in f[layer_name]: # Check using H5 name
                                h5_param_shape = f[layer_name][h5_param_name].shape
                                model_param_shape = vgg_params[layer_name][param_name].shape
                                if h5_param_shape == model_param_shape:
                                    vgg_params[layer_name][param_name] = jnp.array(f[layer_name][h5_param_name]) # Load using H5 name
                                else:
                                    print(f"Warning: Shape mismatch for {layer_name}/{param_name} (H5: {h5_param_name}): H5={h5_param_shape}, Model={model_param_shape}. Skipping.")
                            else:
                                print(f"Warning: Param '{param_name}' (H5: '{h5_param_name}') not found in layer '{layer_name}' in h5 file.")
                    else:
                        print(f"Warning: Layer '{layer_name}' not found in h5 file.")
            vgg_params = flax.core.freeze(vgg_params)
            print("Finished loading local VGG weights.")
        else:
            print("Local VGG weights file not found. Proceeding without pretrained weights.")

        # Transfer the loaded (or initial) weights to the target model's params
        if self.config.training.n_pretrained_layers == 0:
            pass
        elif self.config.training.n_pretrained_layers == 1:
            params["Conv_0"] = vgg_params["conv1_1"]
        elif self.config.training.n_pretrained_layers == 2:
            params["Conv_0"] = vgg_params["conv1_1"]
            params["Conv_1"] = vgg_params["conv1_2"]
        else:
            raise ValueError(
                "Transferring pretrained VGG weights from >2 layers not supported."
            )

        return params

    def create_task_opt_train_state(self, model):
        self.root_key, params_key = random.split(self.root_key, num=2)
        img_h, img_w = self.config.data.img_h, self.config.data.img_w
        imgs = jnp.ones([2, img_h, img_w, 3])

        params = model.init(params_key, imgs, training=False)["params"]
        params = params.unfreeze()
        # Transfer weights from pretrained VGG
        params = self._transfer_vgg_weights(params, imgs)

        cnn_opt = _get_cnn_opt(self.config)
        tx = optax.chain(
            optax.clip(self.config.optimizer.clip_val),
            cnn_opt,
        )

        train_state = TrainState.create(
            apply_fn=model.apply, params=params, tx=tx, key=self.dropout_key
        )

        return train_state

    def create_vam_train_state(self, model):
        self.root_key, params_key, mc_key = random.split(self.root_key, num=3)
        responses = jnp.ones([2]).astype(int)
        rts = jnp.ones([2])
        img_h, img_w = self.config.data.img_h, self.config.data.img_w
        imgs = jnp.ones([2, img_h, img_w, 3])
        # Create placeholder for g with correct shape (dummy_batch_size, n_acc)
        g_placeholder = jnp.ones((2, self.config.model.n_acc))

        params = model.init(params_key, imgs, rts, responses, g_placeholder, mc_key, training=False)[
            "params"
        ]
        params = params.unfreeze()
        # Transfer weights from pretrained VGG
        params["get_drifts"] = self._transfer_vgg_weights(params["get_drifts"], imgs)

        label_fn = flattened_traversal(vam_label_fn)
        cnn_opt = _get_cnn_opt(self.config)
        vi_opt = _get_vi_opt(self.config)
        tx = optax.chain(
            optax.clip(self.config.optimizer.clip_val),
            optax.multi_transform(
                {
                    "cnn": cnn_opt,
                    "vi": vi_opt,
                },
                label_fn,
            ),
        )

        # Ensure params are mutable dict for optax multi_transform init
        mutable_params = flax.core.unfreeze(params)

        train_state = TrainState.create(
            apply_fn=model.apply, params=mutable_params, tx=tx, key=self.dropout_key
        )

        return train_state

    def train(self):
        # Get user metrics once
        if (self.model_type in ["vam", "binned_rt"]) and (self.start_epoch == 0):
            self.root_key, data_key = random.split(self.root_key)
            val_gen = self.task_data.data_generator(data_key, "val", shuffle=False)
            user_metrics = get_vam_user_metrics(val_gen)
        else:
            user_metrics = None

        # Main training loop
        plot_counter = 0
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            self.root_key, data_key = random.split(self.root_key)
            train_gen = self.task_data.data_generator(data_key, "train", shuffle=True)
            val_gen = self.task_data.data_generator(data_key, "val", shuffle=False)
            train_metrics, eval_metrics = None, None

            for batch in train_gen:
                if plot_counter < self.config.training.n_batches_plot:
                    plot_imgs = batch[0][:15]
                    plot_targ_dirs = batch[3][:15]
                    plot_batch_imgs(plot_imgs, plot_targ_dirs)
                    plot_counter += 1

                self.root_key, mc_key = random.split(self.root_key)
                if self.model_type in ["vam", "binned_rt"]:
                    self.state, metrics = vam_train_step(
                        self.state, batch, mc_key, self.dropout_key
                    )
                elif self.model_type == "task_opt":
                    self.state, metrics = task_opt_train_step(
                        self.state, batch, self.dropout_key
                    )

                if train_metrics is None:
                    train_metrics = metrics
                else:
                    train_metrics = train_metrics.merge(metrics)

            # Calculate and log validation metrics
            if ((epoch + 1) % self.config.training.metrics_every == 0) or (
                epoch == self.config.training.n_epochs - 1
            ):
                model_lba_params = None
                for batch in val_gen:
                    if self.model_type in ["vam", "binned_rt"]:
                        (
                            self.state,
                            self.root_key,
                            metrics,
                            model_lba_params,
                        ) = vam_eval_step(
                            self.state,
                            batch,
                            self.root_key,
                        )
                    elif self.model_type == "task_opt":
                        self.state, metrics = task_opt_eval_step(
                            self.state,
                            batch,
                        )

                    if eval_metrics is None:
                        eval_metrics = metrics
                    else:
                        eval_metrics = eval_metrics.merge(metrics)

                train_metrics = train_metrics.compute()
                eval_metrics = eval_metrics.compute()
                log_metrics(
                    train_metrics,
                    eval_metrics,
                    user_metrics,
                    epoch,
                    model_lba_params,
                )

                # Save checkpoint
                ckpt = {"model": self.state}
                checkpoints.save_checkpoint(
                    ckpt_dir=self.expt_ckpt_dir,
                    target=ckpt,
                    step=epoch,
                    overwrite=True,
                    keep_every_n_steps=self.config.training.metrics_every,
                    orbax_checkpointer=self.checkpointer,
                )

                print(f"epoch: {epoch}")
                print(f"val loss: {eval_metrics['loss']}")
                print(f"train loss: {train_metrics['loss']}")
                # print(f"val accuracy: {eval_metrics['accuracy']}")

        wandb.finish()
