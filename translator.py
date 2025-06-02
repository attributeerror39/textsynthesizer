from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-de-en'):
        """
        Initializes the Translator with a pre-trained translation model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained model to use for translation.
        """
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

    def translate(self, text):
        """
        Translates the input text from German to English using the pre-trained model.

        Args:
            text (str): The text to translate.

        Returns:
            str: The translated text.
        """
        # Tokenize input text
        tokenized_text = self.tokenizer(text, return_tensors="pt")

        # Perform translation
        translated = self.model.generate(**tokenized_text)

        # Decode the translated tokens into text
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)

        return translated_text

# Example usage:
if __name__ == "__main__":
    translator = Translator()
    text_to_translate = "Guten Morgen"
    translated_text = translator.translate(text_to_translate)
    print(f"Translated: {translated_text}")