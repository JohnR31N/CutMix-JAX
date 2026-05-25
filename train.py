import argparse
from functools import partial
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from cutmix_jax.datasets import get_cifar10_dataset, numpy_iterator
from cutmix_jax.losses import classification_loss, cutmix_loss
from cutmix_jax.cutmix import cutmix_batch
from cutmix_jax.models.small_cnn import SmallCNN


class TrainState(train_state.TrainState):
    batch_stats: Dict[str, Any]


def parse_args():
    parser = argparse.ArgumentParser(description="Generic JAX training script")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="small_cnn")
    parser.add_argument("--aug", type=str, default="none", choices=["none", "cutmix"])

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=1.0)

    return parser.parse_args()


def create_model(model_name: str, num_classes: int):
    if model_name == "small_cnn":
        return SmallCNN(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def create_datasets(dataset_name: str, batch_size: int, data_dir: str):
    if dataset_name == "cifar10":
        train_ds = get_cifar10_dataset(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            data_dir=data_dir,
        )

        test_ds = get_cifar10_dataset(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            data_dir=data_dir,
        )

        num_classes = 10
        return train_ds, test_ds, num_classes

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_train_state(rng, model, learning_rate: float):
    dummy_x = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    variables = model.init(rng, dummy_x, train=True)

    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


@jax.jit
def train_step_baseline(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    def loss_fn(params):
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }

        logits, new_model_state = state.apply_fn(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
        )

        loss = classification_loss(logits, labels)
        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    metrics = {
        "loss": loss,
        "acc": acc,
    }

    return state, metrics


@partial(jax.jit, static_argnames=("cutmix_alpha",))
def train_step_cutmix(state, batch, rng, cutmix_alpha: float):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    mixed_images, info = cutmix_batch(
        images=images,
        labels=labels,
        rng=rng,
        alpha=cutmix_alpha,
    )

    def loss_fn(params):
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }

        logits, new_model_state = state.apply_fn(
            variables,
            mixed_images,
            train=True,
            mutable=["batch_stats"],
        )

        loss = cutmix_loss(logits, info)
        return loss, (logits, new_model_state)

    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    # CutMix train acc is only a rough reference,
    # because the target is a mixed label.
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == info["labels_a"])

    metrics = {
        "loss": loss,
        "acc": acc,
        "lam": info["lam"],
    }

    return state, metrics


@jax.jit
def eval_step(state, batch):
    images = jnp.asarray(batch["image"])
    labels = jnp.asarray(batch["label"])

    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }

    logits = state.apply_fn(
        variables,
        images,
        train=False,
        mutable=False,
    )

    loss = classification_loss(logits, labels)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    metrics = {
        "loss": loss,
        "acc": acc,
    }

    return metrics


def run_epoch_train(state, train_ds, args, rng):
    train_losses = []
    train_accs = []
    train_lams = []

    cutmix_count = 0
    total_count = 0

    for step, batch in enumerate(numpy_iterator(train_ds)):
        rng, step_rng, prob_rng = jax.random.split(rng, 3)
        total_count += 1

        if args.aug == "none":
            state, metrics = train_step_baseline(state, batch)

        elif args.aug == "cutmix":
            use_cutmix = bool(jax.random.uniform(prob_rng) < args.cutmix_prob)

            if use_cutmix:
                state, metrics = train_step_cutmix(
                    state,
                    batch,
                    step_rng,
                    args.cutmix_alpha,
                )
                train_lams.append(float(metrics["lam"]))
                cutmix_count += 1
            else:
                state, metrics = train_step_baseline(state, batch)

        else:
            raise ValueError(f"Unsupported augmentation: {args.aug}")

        train_losses.append(float(metrics["loss"]))
        train_accs.append(float(metrics["acc"]))

    output = {
        "loss": sum(train_losses) / len(train_losses),
        "acc": sum(train_accs) / len(train_accs),
    }

    if train_lams:
        output["lam"] = sum(train_lams) / len(train_lams)

    if args.aug == "cutmix":
        output["cutmix_rate"] = cutmix_count / total_count

    return state, output, rng


def run_epoch_eval(state, test_ds):
    test_losses = []
    test_accs = []

    for batch in numpy_iterator(test_ds):
        metrics = eval_step(state, batch)
        test_losses.append(float(metrics["loss"]))
        test_accs.append(float(metrics["acc"]))

    return {
        "loss": sum(test_losses) / len(test_losses),
        "acc": sum(test_accs) / len(test_accs),
    }


def main():
    args = parse_args()

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    train_ds, test_ds, num_classes = create_datasets(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
    )

    state = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.lr,
    )

    print("=" * 80)
    print("Training configuration")
    print(f"Dataset:       {args.dataset}")
    print(f"Model:         {args.model}")
    print(f"Augmentation:  {args.aug}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed:          {args.seed}")
    print(f"Data dir:      {args.data_dir}")

    if args.aug == "cutmix":
        print(f"CutMix alpha:  {args.cutmix_alpha}")
        print(f"CutMix prob:   {args.cutmix_prob}")

    print(f"Device:        {jax.devices()}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        state, train_metrics, rng = run_epoch_train(
            state=state,
            train_ds=train_ds,
            args=args,
            rng=rng,
        )

        test_metrics = run_epoch_eval(
            state=state,
            test_ds=test_ds,
        )

        msg = (
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['acc']:.4f} | "
            f"test loss {test_metrics['loss']:.4f} | "
            f"test acc {test_metrics['acc']:.4f}"
        )

        if "lam" in train_metrics:
            msg += f" | avg lam {train_metrics['lam']:.4f}"

        if "cutmix_rate" in train_metrics:
            msg += f" | cutmix rate {train_metrics['cutmix_rate']:.4f}"

        print(msg)


if __name__ == "__main__":
    main()