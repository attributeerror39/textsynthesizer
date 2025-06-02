import spacy
from langdetect import detect
import textstat
import pyphen
import re
import random
import gradio as gr
from sia import SentimentAnalysis
from translator import Translator
translator = Translator()
# from llm import run_llm

''' FUNCTIONS '''

def detect_language(text):
    try:
        language = detect(text)
        return language
    except Exception as e:
        print(e)
        return 'en'


def select_nlp(text):
    lang = detect_language(text)
    if lang == "de":
        return spacy.load("de_core_news_sm")
    else:
        return spacy.load("en_core_web_sm")

def clear_input():
    return ''


def output_to_input(text):
    return text


def split_sentences(text):

    nlp = select_nlp(text)
    doc = nlp(text)  # Create a SpaCy document from the input text
    my_sentences = []  # Initialize an empty list to store sentences
    for sent in doc.sents:  # Iterate through the sentences in the document
        my_sentences.append(str(sent))  # Append each sentence as a string to the "my_sentences" list
    return my_sentences


def extract_words(text):
    nlp = select_nlp(text)
    # Process the input text into a spaCy Doc object
    doc = nlp(text)
    
    # Extract words (tokens) from the Doc object, ignoring punctuation and spaces
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    return words


def shuffle_sentences(text):
    # split sentences
    sentences = split_sentences(text)
    # shuffle sentencers
    random.shuffle(sentences)
    return " ".join(sentences)


def shuffle_nouns(text):
    # language = detect_language(text)
    # if language == 'en':
    #     nlp = nlp_en
    # elif language == 'de':
    #     nlp = nlp_de
    # else:
    #     raise ValueError("Unsupported language. Supported languages are 'en' and 'de'.")
    nlp = select_nlp(text)
    # Tokenize and process the text
    doc = nlp(text)

    # Extract nouns
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']

    # Shuffle the nouns
    shuffle(nouns)

    # Replace the original nouns with the shuffled nouns
    noun_iter = iter(nouns)
    result = []
    noun_index = 0

    for token in doc:
        if token.pos_ == 'NOUN':
            result.append(next(noun_iter))
        else:
            result.append(token.text)

    result = ' '.join(result)
    result = remove_spaces_before_punctuation(result)

    return result


def shuffle_lines(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Shuffle the lines
    random.shuffle(lines)

    # Join the shuffled lines back into a string
    shuffled_string = '<br>'.join(lines)

    return shuffled_string


def remove_spaces_before_punctuation(text):
    # Define a regex pattern to match spaces before punctuation
    pattern = r'(?<=\w) (?=[.,!?;:])'

    # Use re.sub to replace matched spaces with an empty string
    result = re.sub(pattern, '', text)

    return result


def count_words(text):
    # split text into words
    words = text.split(' ')
    # count words
    num_words = len(words)
    return num_words


def reading_ease(text):
    lang = detect_language(text)
    textstat.set_lang(lang)
    return textstat.flesch_reading_ease(text)


def sentiment_score(text):
    sia = SentimentAnalysis()
    score, _, _ = sia.return_score_and_words(text, 5)
    # print(text, score)
    return score


def sort(text, method, reverse=False):
    # split sentences
    sentences = split_sentences(text)
    # select method
    if method == 'alphabetical':
        sentences = sorted(sentences, reverse=reverse)
        return ' '.join(sentences)
    elif method == 'reading ease':
        # flip reverse 
        reverse = not reverse
        sentences = sorted(sentences, key=reading_ease, reverse=reverse)
        return ' '.join(sentences)
    elif method == 'sentiment':
        # flip reverse 
        reverse = not reverse
        sentences = sorted(sentences, key=sentiment_score, reverse=reverse)
        return ' '.join(sentences)
    else:  # word count
        sentences = sorted(sentences, key=count_words, reverse=reverse)
        return '\n\n'.join(sentences)


def do_sentiment_analysis(text):
    try:
        lang = detect_language(text)
        if lang == 'de':
            text = translator.translate(text)
        
        # method 2: from sia
        sia = SentimentAnalysis()
        score, pos_words, neg_words = sia.return_score_and_words(text, 5)
        print('pos:', pos_words)
        print('neg:', neg_words)

        html = highlight_words(text, [pos_words, neg_words], ["background-color: #9fff33; color: black;", "background-color: #ff5733; color: black;"])  # Highlight the first n words.
        html += f"<br><br>score: {score}"
        return html
    except Exception as e:
        return e

def highlight_words(text, words_to_highlight, highlight_style):
    """
    Highlight specific words in a given text and return the HTML code with inline CSS styling.

    :param text: str, the input text to analyze
    :param words_to_highlight: list, the list of words to highlight
    :return: str, the HTML code with highlighted words
    """
    # CSS style for highlighted words
    # highlight_style = "background-color: yellow; color: black;"

    # Tokenize the input text
    # tokenized_text = nltk.word_tokenize(text)

    # Find all word tokens and their positions
    word_pattern = re.compile(r'\b\w+\b')
    matches = word_pattern.finditer(text)

    # Function to check if a word should be highlighted
    def highlight_word(match, words, style):
        word = match.group()
        return f'<span style="{style}">{word}</span>' if word in words else word

    def highlight_words(match, words, styles):
        word = match.group()
        if word in words[0]:
            return f'<span style="{styles[0]}">{word}</span>'
        elif word in words[1]:
            return f'<span style="{styles[1]}">{word}</span>'
        else:
            return word

    # Reconstruct the text with highlighted words
    highlighted_text = word_pattern.sub(lambda match: highlight_words(match, words_to_highlight, highlight_style), text)

    return highlighted_text


def italicize_gerunds(text):
    nlp = select_nlp(text)
    doc = nlp(text)
    output = []

    for token in doc:
        # Check if word ends with 'ing', is a verb, and is a gerund (VBG)
        # if token.text.lower().endswith("ing") and token.tag_ == "VBG":
        if token.tag_ == "VBG":
            output.append(f"<i>{token.text}</i>{token.whitespace_}")
        else:
            output.append(token.text_with_ws)
    
    return ''.join(output)


# Define a function to redact adjectives
def redact_adjectives(text):
    # Process the text using Spacy
    nlp = select_nlp(text)
    doc = nlp(text)

    # Initialize an empty list to store the redacted text
    redacted_text = ""
    black_square_unicode = "\u25A0"
    # Iterate over the tokens in the document
    for token in doc:
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #     token.shape_, token.is_alpha, token.is_stop)
        # Check if the token is an adjective
        if token.pos_ == "ADJ":
            # Redact the adjective by adding the Unicode symbol
            redacted_text += f" {black_square_unicode*len(token.text)}"
            # redacted_text += f"{black_square_unicode*len(token.text)}  "  
        # If the token is a noun or a verb
        else:
            # Add the token to the redacted text
            redacted_text += " " + token.text

    # Remove any leading or trailing whitespace from the redacted text
    redacted_text = redacted_text.strip()
    redacted_text = remove_spaces_before_punctuation(redacted_text)
    return redacted_text


def muenzwurf_baroque(text):
    result = ""
    for i, char in enumerate(text):
        if i % 2 == 0 and char == 'k':
            result += 'c'
        elif i % 2 == 1 and char == 'i':
            result += 'y'
        else:
            result += char
    return result

# HAIKU 
def count_syllables(word, dic):
    syllables = dic.inserted(word)
    return max(1, syllables.count('-') + 1)

def text_to_haikus(text):
    lang = detect_language(text)
    dic = pyphen.Pyphen(lang=lang)
    words = text.split()
    
    haiku_syllable_pattern = [5, 7, 5]
    haikus = []
    current_haiku = []
    current_line = []
    current_syllables = 0
    line_index = 0
    
    for word in words:
        syll_count = count_syllables(word, dic)
        # Check if adding this word exceeds current line syllable count
        if current_syllables + syll_count > haiku_syllable_pattern[line_index]:
            # Finish current line and move to next
            current_haiku.append(' '.join(current_line))
            current_line = [word]
            current_syllables = syll_count
            line_index += 1
            
            # If finished 3 lines, store the haiku and reset
            if line_index >= 3:
                haikus.append('<br>'.join(current_haiku))
                current_haiku = []
                current_line = []
                current_syllables = 0
                line_index = 0
        else:
            current_line.append(word)
            current_syllables += syll_count
    
    # Add any leftover line in current haiku
    if current_line:
        current_haiku.append(' '.join(current_line))
    
    # Pad current haiku if lines are less than 3
    while len(current_haiku) < 3 and len(current_haiku) > 0:
        current_haiku.append('')
    
    # Add the last haiku if it has any content
    if current_haiku:
        haikus.append('<br>'.join(current_haiku))
    
    # Return all haikus separated by a blank line
    return '<br><br>'.join(haikus)


def transform_otoO(text):
    return text.replace("o", "0").replace("O", "0")


''' INTERFACE '''

css = '''
.text {font-size: 1.2rem; !important}
'''

html = gr.HTML(elem_classes="text")

with gr.Blocks(css=css) as demo:

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.TextArea(label="input", elem_classes="text")
            # gr.HTML('output')
            # output = gr.TextArea(label="output", elem_classes="text")
            html.render()
            
        with gr.Column(scale=1):
            
                # clear input
                clear_input_btn = gr.Button("clear input")
                clear_input_btn.click(fn=clear_input, outputs=input_text)    
                # output to input
                output_to_input_btn = gr.Button("output to input")
                output_to_input_btn.click(fn=output_to_input, inputs=html, outputs=input_text)

                # shuffle lines
                shuffle_btn = gr.Button("shuffle lines")
                shuffle_btn.click(fn=shuffle_lines, inputs=input_text, outputs=html)
                # shuffle sentences
                shuffle_btn = gr.Button("shuffle sentences")
                shuffle_btn.click(fn=shuffle_sentences, inputs=input_text, outputs=html)
                # shuffle nouns
                shuffle_nouns_btn = gr.Button("shuffle nouns")
                shuffle_nouns_btn.click(fn=shuffle_nouns, inputs=input_text, outputs=html)

                # with gr.Column():
                sort_method_radio = gr.Radio(["alphabetical", "reading ease", "sentiment", "word count"], label="sort method")
                sort_order_checkbox = gr.Checkbox(label="reverse order", value=False)
                # sort button
                sort_btn = gr.Button("sort")
                sort_btn.click(fn=sort, inputs=[input_text, sort_method_radio, sort_order_checkbox], outputs=html)
                # sentiment analysis
                sentiment_btn = gr.Button("sentiment analysis")
                sentiment_btn.click(fn=do_sentiment_analysis, inputs=input_text, outputs=html)

                # Gerunds -> Italics
                italicize_gerunds_btn = gr.Button("gerunds -> italics")
                italicize_gerunds_btn.click(fn=italicize_gerunds, inputs=input_text, outputs=html)

                # Redact Adjectives
                redact_adjectives_btn = gr.Button("Redact Adjectives")
                redact_adjectives_btn.click(fn=redact_adjectives, inputs=input_text, outputs=html)

                # Münzwurf Baroque
                muenzwurf_baroque_btn = gr.Button("Münzwurf Baroque")
                muenzwurf_baroque_btn.click(fn=muenzwurf_baroque, inputs=input_text, outputs=html)

                # Split into Haiku
                text_to_haikus_btn = gr.Button("Haiku")
                text_to_haikus_btn.click(fn=text_to_haikus, inputs=input_text, outputs=html)

                # Split into Haiku
                transform_otoO_btn = gr.Button("oto0")
                transform_otoO_btn.click(fn=transform_otoO, inputs=input_text, outputs=html)

# demo.launch()
demo.launch(server_port=8898, server_name="0.0.0.0", debug=True)