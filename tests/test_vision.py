import base64
import unittest
from unittest.mock import patch

from vision import (
    VisionError,
    VisionNotConfigured,
    analyze_images,
    validate_image_data_url,
)


class VisionInputTests(unittest.TestCase):
    def test_valid_png_data_url_is_accepted(self):
        # PNG signature plus minimal test payload; only the signature is checked.
        payload = b"\x89PNG\r\n\x1a\n" + b"test-image"
        data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        mime, decoded = validate_image_data_url(data_url)
        self.assertEqual(mime, "image/png")
        self.assertEqual(decoded, payload)

    def test_wrong_signature_is_rejected(self):
        payload = b"not-a-png"
        data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        with self.assertRaisesRegex(VisionError, "не соответствует"):
            validate_image_data_url(data_url)

    def test_remote_url_is_not_accepted_as_image_upload(self):
        with self.assertRaises(VisionError):
            validate_image_data_url("https://example.com/image.png")

    def test_image_analysis_requires_server_side_key(self):
        payload = b"\x89PNG\r\n\x1a\n" + b"test-image"
        data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        with patch.dict("os.environ", {"SWIN_VISION_API_KEY": "", "OPENAI_API_KEY": ""}):
            with self.assertRaises(VisionNotConfigured):
                analyze_images([data_url], "Что на фото?", "ru")

    def test_more_than_three_images_is_rejected(self):
        with self.assertRaisesRegex(VisionError, "не более трёх"):
            analyze_images(["bad"] * 4, "Опиши", "ru")


if __name__ == "__main__":
    unittest.main()
