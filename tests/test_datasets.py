from cutmix_jax.datasets import get_cifar10_dataset, numpy_iterator


def test_cifar10_train_batch_shape():
    ds = get_cifar10_dataset(
        split="train",
        batch_size=8,
        shuffle=False,
        data_dir="./data",
    )

    batch = next(iter(numpy_iterator(ds)))

    assert batch["image"].shape == (8, 32, 32, 3)
    assert batch["label"].shape == (8,)
    assert batch["image"].dtype.name == "float32"
    assert batch["label"].dtype.name in ["int32", "int64"]


def test_cifar10_test_batch_shape():
    ds = get_cifar10_dataset(
        split="test",
        batch_size=8,
        shuffle=False,
        data_dir="./data",
    )

    batch = next(iter(numpy_iterator(ds)))

    assert batch["image"].shape == (8, 32, 32, 3)
    assert batch["label"].shape == (8,)
    assert batch["image"].dtype.name == "float32"