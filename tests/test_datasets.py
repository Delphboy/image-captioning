import importlib
import os
import sys
import unittest
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tests import settings

DATASET_IMPORT_ERROR = None
Flickr8K = None
Coco = None
get_dataset = None
get_dataloader = None

try:
    datasets_module = importlib.import_module("datasets")
    Flickr8K = getattr(datasets_module, "Flickr8K")
    Coco = getattr(datasets_module, "Coco")
    get_dataset = getattr(datasets_module, "get_dataset")
    get_dataloader = getattr(datasets_module, "get_dataloader")
except ModuleNotFoundError as exc:
    DATASET_IMPORT_ERROR = exc


class TestFlickr8kLoading(unittest.TestCase):
    def test_flickr8k_all_split(self) -> None:
        assert Flickr8K is not None
        opts = SimpleNamespace(
            dataset="flickr8k",
            dataset_json=settings.FLICKR8K_DATASET_JSON,
            img_root=settings.FLICKR8K_IMG_ROOT,
            split="all",
        )
        dataset = Flickr8K(opts)

        self.assertEqual(dataset.dataset, "flickr8k")
        self.assertEqual(len(dataset), 8000)

        first = dataset[0]
        self.assertIn("img", first)
        self.assertIn("captions", first)
        self.assertEqual(len(first["captions"]), 5)

    def test_flickr8k_train_split(self) -> None:
        assert Flickr8K is not None
        opts = SimpleNamespace(
            dataset="flickr8k",
            dataset_json=settings.FLICKR8K_DATASET_JSON,
            img_root=settings.FLICKR8K_IMG_ROOT,
            split="train",
        )
        dataset = Flickr8K(opts)

        self.assertGreater(len(dataset), 0)
        self.assertLess(len(dataset), 8000)
        self.assertTrue(all(image["split"] == "train" for image in dataset.images))

class TestCocoLoading(unittest.TestCase):
    def test_coco_all_split(self) -> None:
        assert Coco is not None
        opts = SimpleNamespace(
            dataset="coco",
            dataset_json=settings.COCO_DATASET_JSON,
            img_root=settings.COCO_IMG_ROOT,
            split="all",
        )
        dataset = Coco(opts)

        self.assertEqual(dataset.dataset, "coco")
        self.assertEqual(len(dataset), 123287)

        first = dataset[0]
        self.assertIn("img", first)
        self.assertIn("captions", first)
        self.assertEqual(len(first["captions"]), 5)

    def test_coco_train_split(self) -> None:
        assert Coco is not None
        opts = SimpleNamespace(
            dataset="coco",
            dataset_json=settings.COCO_DATASET_JSON,
            img_root=settings.COCO_IMG_ROOT,
            split="train",
        )
        dataset = Coco(opts)

        self.assertEqual(len(dataset), 113287)
        self.assertTrue(all(image["split"] in ["train", "restval"] for image in dataset.images))

    def test_coco_val_split(self) -> None:
        assert Coco is not None
        opts = SimpleNamespace(
            dataset="coco",
            dataset_json=settings.COCO_DATASET_JSON,
            img_root=settings.COCO_IMG_ROOT,
            split="val",
        )
        dataset = Coco(opts)

        self.assertEqual(len(dataset), 5000)
        self.assertTrue(all(image["split"] == "val" for image in dataset.images))

    def test_coco_test_split(self) -> None:
        assert Coco is not None
        opts = SimpleNamespace(
            dataset="coco",
            dataset_json=settings.COCO_DATASET_JSON,
            img_root=settings.COCO_IMG_ROOT,
            split="test",
        )
        dataset = Coco(opts)

        self.assertEqual(len(dataset), 5000)
        self.assertTrue(all(image["split"] == "test" for image in dataset.images))


class TestDatasetFactory(unittest.TestCase):
    def test_get_dataset_flickr8k(self) -> None:
        assert get_dataset is not None
        assert Flickr8K is not None
        opts = SimpleNamespace(
            dataset="flickr8k",
            dataset_json=settings.FLICKR8K_DATASET_JSON,
            img_root=settings.FLICKR8K_IMG_ROOT,
            split="train",
        )

        dataset = get_dataset(opts)
        self.assertIsInstance(dataset, Flickr8K)
        self.assertGreater(len(dataset), 0)

    def test_get_dataset_coco(self) -> None:
        assert get_dataset is not None
        assert Coco is not None
        opts = SimpleNamespace(
            dataset="coco",
            dataset_json=settings.COCO_DATASET_JSON,
            img_root=settings.COCO_IMG_ROOT,
            split="train",
        )

        dataset = get_dataset(opts)
        self.assertIsInstance(dataset, Coco)
        self.assertGreater(len(dataset), 0)

    def test_get_dataset_invalid_name(self) -> None:
        assert get_dataset is not None
        opts = SimpleNamespace(
            dataset="unknown",
            dataset_json=settings.FLICKR8K_DATASET_JSON,
            img_root=settings.FLICKR8K_IMG_ROOT,
            split="all",
        )

        with self.assertRaises(ValueError):
            get_dataset(opts)


class TestDataloaderFactory(unittest.TestCase):
    def test_get_dataloader_flickr8k_batch_shape(self) -> None:
        assert get_dataloader is not None
        opts = SimpleNamespace(
            dataset="flickr8k",
            dataset_json=settings.FLICKR8K_DATASET_JSON,
            img_root=settings.FLICKR8K_IMG_ROOT,
            split="train",
            batch_size=8,
            image_size=224,
        )

        dataloader = get_dataloader(opts)
        batch = next(iter(dataloader))

        self.assertIn("img", batch)
        self.assertIn("captions", batch)
        self.assertEqual(tuple(batch["img"].shape), (8, 3, 224, 224))
        self.assertEqual(len(batch["captions"]), 8)

    def test_get_dataloader_coco_batch_shape(self) -> None:
        assert get_dataloader is not None
        opts = SimpleNamespace(
            dataset="coco",
            dataset_json=settings.COCO_DATASET_JSON,
            img_root=settings.COCO_IMG_ROOT,
            split="train",
            batch_size=8,
            image_size=224,
        )

        dataloader = get_dataloader(opts)
        batch = next(iter(dataloader))

        self.assertIn("img", batch)
        self.assertIn("captions", batch)
        self.assertEqual(tuple(batch["img"].shape), (8, 3, 224, 224))
        self.assertEqual(len(batch["captions"]), 8)


if __name__ == "__main__":
    unittest.main()
