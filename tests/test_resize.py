"""The exact area downscale behind [H3] Output scale."""

import unittest
from fractions import Fraction

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class AreaResizeTests(unittest.TestCase):
    def test_whole_factors_are_block_means(self):
        from forge_h3.resize import area_resize
        image = torch.rand(1, 3, 64, 96)
        for scale, block in ((Fraction(1, 2), 2), (Fraction(1, 4), 4)):
            with self.subTest(scale=scale):
                torch.testing.assert_close(area_resize(image, scale), torch.nn.functional.avg_pool2d(image, block))

    def test_three_quarters_is_the_exact_area_mean(self):
        from forge_h3.resize import area_resize
        image = torch.rand(1, 3, 64, 96, dtype=torch.float64)
        # exact reference: each source pixel split into 3x3, then 4x4 blocks averaged
        fine = image.repeat_interleave(3, -2).repeat_interleave(3, -1)
        reference = torch.nn.functional.avg_pool2d(fine, 4)
        torch.testing.assert_close(area_resize(image, Fraction(3, 4)), reference)

    def test_two_thirds_rounds_a_side_that_does_not_divide(self):
        from forge_h3.resize import area_matrix, area_resize, scaled_size
        self.assertEqual((scaled_size(4096, Fraction(2, 3)), scaled_size(4096, Fraction(3, 4))), (2731, 3072))
        image = torch.rand(1, 3, 32, 64)
        out = area_resize(image, Fraction(2, 3))
        self.assertEqual(tuple(out.shape), (1, 3, 21, 43))
        # every output pixel is a weighted mean: rows sum to 1, so a flat picture stays flat and the mean is kept
        matrix = area_matrix(4096, 2731, torch.float64)
        torch.testing.assert_close(matrix.sum(1), torch.ones(2731, dtype=torch.float64))
        torch.testing.assert_close(area_resize(torch.full((1, 3, 32, 64), 0.4), Fraction(2, 3)),
                                   torch.full((1, 3, 21, 43), 0.4))
        self.assertAlmostEqual(float(out.mean()), float(image.mean()), places=2)

    def test_scale_one_returns_the_picture(self):
        from forge_h3.resize import area_resize
        image = torch.rand(1, 3, 8, 8)
        self.assertIs(area_resize(image, Fraction(1)), image)


if __name__ == "__main__":
    unittest.main()
