import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from forge_h3.contracts import H3Error
from forge_h3.integration import ProcessingRouter, validate_processing


class RoutingTests(unittest.TestCase):
    def test_non_h3_goes_to_original_without_hooks(self):
        original = Mock(return_value="Forge")
        generate = Mock()
        router = ProcessingRouter(original, lambda p: None, generate)
        p = SimpleNamespace(scripts=Mock())
        self.assertEqual(router(p), "Forge")
        original.assert_called_once_with(p)
        generate.assert_not_called()
        p.scripts.before_process.assert_not_called()

    def test_h3_routes_before_native_model_loader(self):
        original = Mock(side_effect=AssertionError("unsupported loader"))
        generate = Mock(return_value="H3")
        selected = object()
        router = ProcessingRouter(original, lambda p: selected, generate)
        p = SimpleNamespace(scripts=Mock())
        self.assertEqual(router(p), "H3")
        original.assert_not_called()
        p.scripts.before_process.assert_called_once_with(p)
        generate.assert_called_once_with(p, selected)

    def test_unsupported_edit_features_fail_before_inference(self):
        p = SimpleNamespace(n_iter=1, enable_hr=True)
        with self.assertRaisesRegex(H3Error, "Hires"):
            validate_processing(p)
        p = SimpleNamespace(n_iter=1, image_mask=object())
        with self.assertRaisesRegex(H3Error, "inpainting"):
            validate_processing(p)

    def test_batch_count_is_not_silently_ignored(self):
        with self.assertRaisesRegex(H3Error, "Batch Count"):
            validate_processing(SimpleNamespace(n_iter=2))

    def test_native_disabled_seed_resize_sentinels_are_accepted(self):
        validate_processing(SimpleNamespace(n_iter=1, subseed_strength=0,
                                            seed_resize_from_w=-1, seed_resize_from_h=-1))

    def test_positive_seed_resize_is_rejected(self):
        with self.assertRaisesRegex(H3Error, "seed resize"):
            validate_processing(SimpleNamespace(n_iter=1, subseed_strength=0,
                                                seed_resize_from_w=512, seed_resize_from_h=512))


if __name__ == "__main__":
    unittest.main()
