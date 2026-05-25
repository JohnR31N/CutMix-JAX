import jax.numpy as jnp

from cutmix_jax.losses import cross_entropy_with_integer_labels, cutmix_loss


def test_cross_entropy_shape():
    logits = jnp.array([
        [2.0, 0.1, 0.1],
        [0.1, 2.0, 0.1],
    ])
    labels = jnp.array([0, 1])

    loss = cross_entropy_with_integer_labels(logits, labels)

    assert loss.shape == (2,)
    assert jnp.all(loss >= 0.0)


def test_cutmix_loss_scalar():
    logits = jnp.array([
        [2.0, 0.1, 0.1],
        [0.1, 2.0, 0.1],
    ])
    info = {
        "labels_a": jnp.array([0, 1]),
        "labels_b": jnp.array([1, 2]),
        "lam": 0.7,
    }

    loss = cutmix_loss(logits, info)

    assert loss.shape == ()
    assert loss >= 0.0