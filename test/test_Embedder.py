import unittest
from src.Embedder import Embedder

class TestEmbedder(unittest.TestCase):

    def test_encode_decode_no_changes(self):
        # Create an instance of the Embedder class
        text = "Hello, world!"
        embedder = Embedder(model="simple", text=text)

        # Define a sample text
        text = "Hello, world!"

        # Encode the text
        encoded_text = embedder.encode(text)

        # Decode the encoded text
        decoded_text = embedder.decode(encoded_text)

        # Check if the decoded text is the same as the original text
        self.assertEqual(decoded_text, text, "Decoding the encoded text should lead to no changes")

