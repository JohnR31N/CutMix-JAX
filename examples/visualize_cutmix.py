import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cutmix_jax.cutmix import cutmix_batch


def make_dummy_images(batch_size=8, height=32, width=32, channels=3):
    images = []

    for i in range(batch_size):
        img = jnp.zeros((height, width, channels))

        # 每张图给一个不同亮度/图案，方便看 patch 是否替换
        value = (i + 1) / batch_size
        img = img.at[:, :, 0].set(value)
        img = img.at[:, :, 1].set(jnp.linspace(0, 1, width)[None, :])
        img = img.at[:, :, 2].set(jnp.linspace(0, 1, height)[:, None])

        images.append(img)

    return jnp.stack(images)


def main():
    rng = jax.random.PRNGKey(0)

    images = make_dummy_images()
    labels = jnp.arange(images.shape[0])

    mixed_images, info = cutmix_batch(images, labels, rng, alpha=1.0)

    print("lambda:", float(info["lam"]))
    print("perm:", info["perm"])
    print("box:", info["box"])

    idx = 0
    source_idx = int(info["perm"][idx])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(images[idx])
    axes[0].set_title(f"Original A: {idx}")
    axes[0].axis("off")

    axes[1].imshow(images[source_idx])
    axes[1].set_title(f"Original B: {source_idx}")
    axes[1].axis("off")

    axes[2].imshow(mixed_images[idx])
    axes[2].set_title("CutMix")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("cutmix_visualization.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()