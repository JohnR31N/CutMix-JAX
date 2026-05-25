import jax.numpy as jnp
import jax.nn as jnn


def cross_entropy_with_integer_labels(logits, labels):
    log_probs = jnn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(labels.shape[0]), labels]


def cutmix_loss(logits, info):
    labels_a = info["labels_a"]
    labels_b = info["labels_b"]
    lam = info["lam"]

    loss_a = cross_entropy_with_integer_labels(logits, labels_a)
    loss_b = cross_entropy_with_integer_labels(logits, labels_b)

    loss = lam * loss_a + (1.0 - lam) * loss_b
    return jnp.mean(loss)