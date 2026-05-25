import jax
import jax.numpy as jnp

from cutmix_jax.models.small_cnn import SmallCNN


def test_small_cnn_output_shape():
    rng = jax.random.PRNGKey(0)
    model = SmallCNN(num_classes=10)

    x = jnp.ones((4, 32, 32, 3))
    variables = model.init(rng, x, train=True)

    logits = model.apply(variables, x, train=False)

    assert logits.shape == (4, 10)