"""
Tests for PetPulse Vision (vision.py, food_checker.py, the label agent call, and the scanner UI).

Images are generated in the test. OCR and Claude are mocked (one optional test
uses the real local Tesseract if it's installed), so no paid API calls are made.
Run with:  python -m unittest -v
"""

import base64
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anthropic
import httpx2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import agent
import database as db
import food_checker as fc
import vision
from test_agent import FakeClient, api_error

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
LABEL = ("TASTY BITES Dog Treats\nINGREDIENTS: Chicken meal, brown rice, oatmeal,\n"
         "chicken fat (preserved with mixed tocopherols),\ndried beet pulp, natural flavor, salt.")


def label_image(text=LABEL, size=(1200, 600)):
    image = Image.new("RGB", size, (245, 240, 225))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 28) if os.path.exists(FONT_PATH) else ImageFont.load_default()
    for row, line in enumerate(text.split("\n")):
        draw.text((50, 40 + row * 40), line, fill=(30, 30, 30), font=font)
    return image


def to_bytes(image, image_format="JPEG", **options):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_valid_jpeg_and_png(self):
        for image_format in ("JPEG", "PNG"):
            image = vision.load_image(to_bytes(label_image(), image_format))
            self.assertEqual((image.mode, image.size), ("RGB", (1200, 600)))

    def test_not_an_image(self):
        for data in (b"", b"not an image at all", b"%PDF-1.4 fake document"):
            with self.assertRaises(vision.VisionError):
                vision.load_image(data)

    def test_unsupported_format_even_if_renamed(self):
        gif = to_bytes(label_image(), "GIF")  # a GIF uploaded as "label.jpg" is still a GIF
        with self.assertRaises(vision.VisionError) as caught:
            vision.load_image(gif)
        self.assertIn("JPG or PNG", str(caught.exception))

    def test_corrupted_image(self):
        data = to_bytes(label_image())
        with self.assertRaises(vision.VisionError):
            vision.load_image(data[: len(data) // 2])  # cut-off download

    def test_oversized_upload(self):
        with mock.patch.object(vision, "MAX_UPLOAD_BYTES", 1000):
            with self.assertRaises(vision.VisionError) as caught:
                vision.load_image(to_bytes(label_image()))
        self.assertIn("too large", str(caught.exception))

    def test_too_many_pixels(self):
        with mock.patch.object(vision, "MAX_PIXELS", 500_000):
            with self.assertRaises(vision.VisionError):
                vision.load_image(to_bytes(label_image()))

    def test_too_small(self):
        with self.assertRaises(vision.VisionError):
            vision.load_image(to_bytes(Image.new("RGB", (150, 100), "white")))

    def test_transparent_png_gets_white_background(self):
        image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
        loaded = vision.load_image(to_bytes(image, "PNG"))
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.getpixel((10, 10)), (255, 255, 255))

    def test_orientation_is_corrected(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90° clockwise to display", as phones write it
        data = to_bytes(Image.new("RGB", (400, 250), "white"), "JPEG", exif=exif.tobytes())
        self.assertEqual(vision.load_image(data).size, (250, 400))


class PreprocessingTests(unittest.TestCase):
    def test_resize_keeps_aspect_ratio(self):
        resized = vision.resize_to_fit(Image.new("RGB", (4000, 3000)), 2400)
        self.assertEqual(resized.size, (2400, 1800))

    def test_resize_never_enlarges(self):
        self.assertEqual(vision.resize_to_fit(Image.new("RGB", (800, 600)), 2400).size, (800, 600))

    def test_blur_detection(self):
        sharp = vision.prepare_label_image(to_bytes(label_image()))
        blurry = vision.prepare_label_image(to_bytes(label_image().filter(ImageFilter.GaussianBlur(4))))
        self.assertFalse(sharp.is_blurry)
        self.assertTrue(blurry.is_blurry)
        self.assertGreater(sharp.blur_score, blurry.blur_score)

    def test_ocr_copy_is_enhanced_grayscale(self):
        prepared = vision.prepare_label_image(to_bytes(label_image()))
        self.assertEqual(prepared.ocr_image.ndim, 2)  # grayscale
        self.assertEqual(prepared.ocr_image.dtype, np.uint8)
        self.assertEqual(prepared.ocr_image.shape, (1200, 2400))  # small images are enlarged 2x for OCR
        self.assertEqual(prepared.color.mode, "RGB")  # color copy kept for Claude

    def test_image_for_claude(self):
        data, media_type = vision.image_for_claude(Image.new("RGB", (3000, 2000), "white"))
        self.assertEqual(media_type, "image/jpeg")
        self.assertEqual(max(Image.open(io.BytesIO(data)).size), vision.CLAUDE_MAX_SIDE)


class OcrTests(unittest.TestCase):
    def fake_data(self, words, confidences):
        count = len(words)
        return {"text": words, "conf": confidences, "block_num": [1] * count, "par_num": [1] * count,
                "line_num": [1 if i < 3 else 2 for i in range(count)]}

    def test_text_lines_and_confidence(self):
        data = self.fake_data(["INGREDIENTS:", "Chicken", "meal,", "", "rice", "salt."], [95, 90, 85, -1, 80, 70])
        with mock.patch("pytesseract.image_to_data", return_value=data):
            result = vision.extract_ingredients_text(np.zeros((10, 10), np.uint8))
        self.assertEqual(result.text, "INGREDIENTS: Chicken meal,\nrice salt.")
        self.assertEqual(result.word_count, 5)
        self.assertEqual(result.confidence, 84.0)
        self.assertTrue(result.readable)

    def test_little_text_is_not_readable(self):
        with mock.patch("pytesseract.image_to_data", return_value=self.fake_data(["~~", "ii"], [20, 10])):
            self.assertFalse(vision.extract_ingredients_text(np.zeros((10, 10), np.uint8)).readable)

    def test_tesseract_not_installed(self):
        import pytesseract

        with mock.patch("pytesseract.image_to_data", side_effect=pytesseract.TesseractNotFoundError()):
            with self.assertRaises(vision.VisionError) as caught:
                vision.extract_ingredients_text(np.zeros((10, 10), np.uint8))
        self.assertIn("brew install tesseract", str(caught.exception))

    def test_ocr_error(self):
        import pytesseract

        with mock.patch("pytesseract.image_to_data", side_effect=pytesseract.TesseractError(1, "boom")):
            with self.assertRaises(vision.VisionError):
                vision.extract_ingredients_text(np.zeros((10, 10), np.uint8))

    @unittest.skipUnless(shutil.which("tesseract") and os.path.exists(FONT_PATH), "needs local Tesseract and Arial")
    def test_real_local_ocr(self):
        prepared = vision.prepare_label_image(to_bytes(label_image()))
        result = vision.extract_ingredients_text(prepared.ocr_image)
        self.assertIn("chicken meal", result.text.lower())
        self.assertTrue(result.readable)


class AllergyMatchingTests(unittest.TestCase):
    def test_chicken_and_chicken_meal(self):
        result = fc.check_allergens("Ingredients: chicken meal, rice, chicken.", "Chicken")
        self.assertEqual(result["status"], fc.POTENTIAL_CONFLICT)
        self.assertEqual([m["phrase"] for m in result["matches"]], ["chicken meal", "chicken"])

    def test_case_and_punctuation_do_not_matter(self):
        result = fc.check_allergens("INGREDIENTS: CHICKEN-FAT; Beef Broth, Chicken By-Products.", "chicken, BEEF")
        phrases = [m["phrase"] for m in result["matches"]]
        self.assertEqual(phrases, ["chicken-fat", "beef broth", "chicken by-products"])

    def test_no_false_matches_from_substrings(self):
        text = "Ingredients: chickpeas, eggplant, beefsteak tomato, graham flour, codfish-free broth, scornful."
        result = fc.check_allergens(text, "Chicken, eggs, beef, pork, corn")
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["status"], fc.NO_MATCH_DETECTED)

    def test_word_split_across_lines(self):
        result = fc.check_allergens("Ingredients: rice, chick-\nen liver, salt.", "Chicken")
        self.assertEqual(result["matches"][0]["phrase"], "chicken liver")

    def test_structured_result(self):
        result = fc.check_allergens("Ingredients: lamb, rice, chicken fat (preserved), salt.", "Chicken")
        [match] = result["matches"]
        self.assertEqual(match, {"allergy": "chicken", "source": "Owner-recorded allergy",
                                 "phrase": "chicken fat", "evidence": "chicken fat"})
        self.assertEqual(set(result), {"status", "restrictions", "matches", "reasons"})

    def test_no_match_never_says_safe(self):
        result = fc.check_allergens("Ingredients: lamb, rice, pumpkin, salt.", "Chicken")
        self.assertEqual(result["status"], fc.NO_MATCH_DETECTED)
        self.assertIn("does not mean the food is safe", result["reasons"][0])

    def test_missing_allergies(self):
        for allergies in (None, "", "None", "Not recorded by the owner"):
            result = fc.check_allergens("Ingredients: lamb, rice, pumpkin, salt.", allergies)
            self.assertEqual(result["status"], fc.UNABLE_TO_VERIFY)

    def test_incomplete_text(self):
        self.assertEqual(fc.check_allergens("lamb", "Chicken")["status"], fc.UNABLE_TO_VERIFY)
        result = fc.check_allergens("lamb, rice, pumpkin, salt", "Chicken")
        self.assertTrue(any("ingredients" in reason for reason in result["reasons"]))  # list may be cut off

    def test_poor_quality_cannot_give_no_match(self):
        result = fc.check_allergens("Ingredients: lamb, rice, pumpkin, salt.", "Chicken", reliable=False)
        self.assertEqual(result["status"], fc.UNABLE_TO_VERIFY)

    def test_conflict_is_reported_even_with_poor_quality(self):
        result = fc.check_allergens("chicken meal", "Chicken", reliable=False)
        self.assertEqual(result["status"], fc.POTENTIAL_CONFLICT)

    def test_diet_restrictions_count(self):
        result = fc.check_allergens("Ingredients: chicken, rice, salt, carrots.", None, diet="Dry dog food without chicken")
        self.assertEqual(result["status"], fc.POTENTIAL_CONFLICT)
        self.assertEqual(result["matches"][0]["source"], "Diet restriction")

    def test_owner_wording_is_understood(self):
        terms = [r["term"] for r in fc.parse_restrictions("Allergic to milk; mild pumpkin intolerance", "Grain-free, no wheat")]
        self.assertEqual(terms, ["dairy", "pumpkin", "wheat", "grain"])


class LabelAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.milo = db.get_pet(db.create_pet({**db.DEMO_PET}))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat", "allergies": "Fish"}))
        self.client = FakeClient(reply="Milo should not eat this: it contains chicken meal.")
        for patcher in (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""}),
            mock.patch("agent.print", create=True),
            mock.patch.object(agent, "_create_client", return_value=self.client),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.image, self.media_type = vision.image_for_claude(label_image())
        self.check = fc.check_allergens(LABEL, self.milo["allergies"], self.milo["diet"])

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def ask(self, pet=None, question="Can Milo eat these treats?", history=()):
        pet = pet or self.milo
        return agent.ask_about_label(pet["id"], self.image, self.media_type, LABEL, self.check, question, history)

    def test_image_text_and_rule_result_are_sent(self):
        self.assertIn("chicken meal", self.ask())
        [request] = self.client.requests
        [message] = request["messages"]
        image_block, label_block, question_block = message["content"]
        self.assertEqual(image_block["type"], "image")
        self.assertEqual(image_block["source"]["media_type"], "image/jpeg")
        self.assertEqual(base64.b64decode(image_block["source"]["data"]), self.image)
        self.assertIn("POTENTIAL_CONFLICT", label_block["text"])
        self.assertIn("Chicken meal, brown rice", label_block["text"])
        self.assertEqual(question_block["text"], "Can Milo eat these treats?")
        self.assertIn("Food label scans", request["system"])
        self.assertIn("never call the food safe", request["system"])
        self.assertEqual(request["model"], "claude-haiku-5-5")  # the same configured model as chat

    def test_only_selected_pets_profile_is_sent(self):
        self.ask(pet=self.luna, question="Can Luna eat these treats?")
        system = self.client.requests[0]["system"]
        self.assertIn('"Luna"', system)
        self.assertNotIn("Golden Retriever", system)

    def test_follow_up_keeps_image_and_conversation(self):
        first = self.ask()
        history = [{"role": "user", "content": "Can Milo eat these treats?"}, {"role": "assistant", "content": first}]
        self.ask(question="What about the chicken fat?", history=history)
        messages = self.client.requests[1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"][2]["text"], "Can Milo eat these treats?")
        self.assertEqual(messages[2]["content"], "What about the chicken fat?")

    def test_image_rejected(self):
        self.client.error = api_error(anthropic.BadRequestError, 400)
        with self.assertRaises(agent.AgentError) as caught:
            self.ask()
        self.assertIn("couldn't analyze this image", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_api_failures_are_friendly(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        for error in (api_error(anthropic.RateLimitError, 429), anthropic.APIConnectionError(request=request)):
            self.client.error = error
            with self.assertRaises(agent.AgentError):
                self.ask()

    def test_empty_answer(self):
        self.client.reply = " "
        with self.assertRaises(agent.AgentError):
            self.ask()


class ScannerAppTests(unittest.TestCase):
    """The Food & Allergy Scanner in the real Streamlit app, with OCR and Claude mocked."""

    def setUp(self):
        from streamlit.testing.v1 import AppTest

        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.milo = db.get_pet(db.create_pet({**db.DEMO_PET}))
        self.luna = db.get_pet(db.create_pet({"name": "Luna", "species": "Cat", "allergies": "Fish"}))
        self.client = FakeClient(reply="Milo should not eat this: it contains chicken meal.")
        self.ocr = mock.Mock(return_value=vision.OcrResult(text=LABEL, confidence=94.0, word_count=30))
        for patcher in (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL": ""}),
            mock.patch("agent.print", create=True),
            mock.patch.object(agent, "_create_client", return_value=self.client),
            mock.patch.object(vision, "extract_ingredients_text", self.ocr),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.at = AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=30).run()
        self.select(self.milo)

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def select(self, pet):
        self.at.selectbox(key="sitter_pet_id").set_value(pet["id"]).run()

    def upload(self, data, pet=None):
        pet = pet or self.milo
        self.at.file_uploader(key=f"scan_upload_{pet['id']}_0").set_value(("label.jpg", data, "image/jpeg")).run()

    def analyze(self, pet=None):
        pet = pet or self.milo
        self.at.button(key=f"scan_analyze_{pet['id']}").click().run()
        self.assertFalse(self.at.exception)

    def ask(self):
        button = next(b for b in self.at.button if b.label == "Ask PetPulse about this label")
        button.click().run()
        self.assertFalse(self.at.exception)

    def errors(self):
        return [e.value for e in self.at.error]

    def test_conflict_flow(self):
        self.upload(to_bytes(label_image()))
        self.assertEqual(self.ocr.call_count, 0)  # nothing runs before Analyze is clicked
        self.analyze()
        self.assertEqual(self.ocr.call_count, 1)
        box = next(t for t in self.at.text_area if (t.key or "").startswith("scan_text_"))
        self.assertEqual(box.value, LABEL)
        self.assertTrue(any("Potential Conflict" in e for e in self.errors()))
        self.assertTrue(any("chicken meal" in m.value for m in self.at.markdown))
        self.assertEqual(self.client.requests, [])  # Claude only on request

        self.ask()
        self.assertEqual(len(self.client.requests), 1)
        self.assertTrue(any("should not eat this" in m.value for m in self.at.markdown))
        self.at.run()
        self.at.run()
        self.assertEqual(len(self.client.requests), 1)  # reruns don't call Claude again
        self.assertEqual(self.ocr.call_count, 1)        # ...or OCR

    def test_claude_failure_keeps_rule_based_result(self):
        self.client.error = api_error(anthropic.BadRequestError, 400)
        self.upload(to_bytes(label_image()))
        self.analyze()
        self.ask()
        errors = self.errors()
        self.assertTrue(any("Potential Conflict" in e for e in errors))
        self.assertTrue(any("couldn't analyze this image" in e for e in errors))

    def test_blurry_photo_is_unable_to_verify(self):
        self.ocr.return_value = vision.OcrResult(text="Ingredients: lamb, rice, pumpkin, salt.", confidence=90.0,
                                                 word_count=6)
        self.upload(to_bytes(label_image().filter(ImageFilter.GaussianBlur(4))))
        self.analyze()
        self.assertTrue(any("blurry" in w.value for w in self.at.warning))
        self.assertTrue(any("Unable to Verify" in w.value for w in self.at.warning))

    def test_invalid_file(self):
        self.upload(b"definitely not an image")
        self.analyze()
        self.assertTrue(any("isn't a valid image" in e for e in self.errors()))
        self.assertEqual(self.ocr.call_count, 0)

    def test_switching_pets_discards_scan(self):
        self.upload(to_bytes(label_image()))
        self.analyze()
        self.assertIsNotNone(self.at.session_state["scan"])
        self.select(self.luna)
        self.assertIsNone(self.at.session_state["scan"])
        self.assertFalse(any("Potential Conflict" in e for e in self.errors()))  # Milo's result isn't shown
        self.assertFalse(any((t.key or "").startswith("scan_text_") for t in self.at.text_area))
        self.select(self.milo)
        self.assertIsNone(self.at.session_state["scan"])  # a new analysis is needed
        self.assertEqual(self.client.requests, [])


if __name__ == "__main__":
    unittest.main()
