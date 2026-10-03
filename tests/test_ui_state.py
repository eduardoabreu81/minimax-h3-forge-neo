import unittest

from forge_h3.ui_state import frame_view, preset_frame_view


class FrameViewTests(unittest.TestCase):
    def test_video_uses_native_frames_grid(self):
        view = frame_view(True, "Video", 124)
        self.assertEqual((view["minimum"], view["maximum"], view["step"]), (5, 362, 17))
        self.assertEqual(view["label"], "Frames")
        self.assertTrue(view["visible"])

    def test_switching_still_hides_native_frame_control(self):
        self.assertFalse(frame_view(True, "Still image", 124)["visible"])

    def test_stale_wan_frames_snap_to_h3_grid(self):
        self.assertEqual(frame_view(True, "Video", 121)["value"], 124)

    def test_regular_image_mode_restores_native_range(self):
        saved = dict(minimum=1, maximum=8, step=1, value=3, label="Batch Size", visible=True)
        self.assertEqual(frame_view(False, "Video", 124, saved), saved)

    def test_native_video_preset_range_is_restored(self):
        view = preset_frame_view(16, 129)
        self.assertEqual((view["maximum"], view["step"], view["label"], view["value"]),
                         (241, 16, "Frames", 129))


if __name__ == "__main__":
    unittest.main()
