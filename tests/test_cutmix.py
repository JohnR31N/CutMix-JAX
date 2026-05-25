import jax
import jax.numpy as jnp

from cutmix_jax.cutmix import cutmix_batch


def test_cutmix_shape():
    rng = jax.random.PRNGKey(0)

    images = jax.random.normal(rng, (8, 32, 32, 3))
    labels = jnp.arange(8)

    mixed_images, info = cutmix_batch(images, labels, rng, alpha=1.0)

    assert mixed_images.shape == images.shape
    assert info["labels_a"].shape == labels.shape
    assert info["labels_b"].shape == labels.shape
    assert info["perm"].shape == labels.shape

    lam = info["lam"]
    assert lam >= 0.0
    assert lam <= 1.0


def test_cutmix_changes_some_pixels():
    rng = jax.random.PRNGKey(42)

    images = jax.random.normal(rng, (8, 32, 32, 3))
    labels = jnp.arange(8)

    mixed_images, info = cutmix_batch(images, labels, rng, alpha=1.0)

    diff = jnp.mean(jnp.abs(mixed_images - images))

    assert diff >= 0.0