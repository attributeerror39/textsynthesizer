import nltk
# Ensure necessary NLTK resources are downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class SentimentAnalysis:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, input_string):
        # Tokenize the input string
        tokens = word_tokenize(input_string)

        # Analyze sentiment
        sentiment_scores = self.sia.polarity_scores(' '.join(tokens))
        compound_score = sentiment_scores['compound']
        positive_score = sentiment_scores['pos']
        negative_score = sentiment_scores['neg']
        neutral_score = sentiment_scores['neu']

        # Determine sentiment
        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'sentiment': sentiment,
            'compound_score': compound_score,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score
        }

        
    def analyze_word_sentiment(self, input_string):
        # Tokenize the input string
        tokens = word_tokenize(input_string)
        # stop_words = set(stopwords.words('english'))
        # filtered_words = [word for word in tokens if word.lower() not in stop_words]
        # print(filtered_words)
        # Analyze sentiment for each word
        sentiment_results = {}
        for token in tokens:
            print(token)
            sentiment_scores = self.sia.polarity_scores(token)
            compound_score = sentiment_scores['compound']
            sentiment_results[token] = {
                'compound_score': compound_score
            }

        return sentiment_results

    def sort_words_by_compound_distance(self, input_string, overall_compound_score):
        # Analyze sentiment for each word
        sentiment_results = self.analyze_word_sentiment(input_string)


        # Sort words by compound distance
        sorted_words = sorted(sentiment_results.items(), key=lambda x: abs(x[1]['compound_score'] - overall_compound_score))

        return sorted_words

    def return_score_and_words(self, input_string, num_words):
        
        overall_sentiment = self.analyze_sentiment(input_string)
        overall_compound_score = overall_sentiment['compound_score']
        sorted_words = self.sort_words_by_compound_distance(input_string, overall_compound_score)
        print(sorted_words)

        
        words_to_highlight = [word for word, result in sorted_words]
        # Reduce to nearest 5
        words_to_highlight = words_to_highlight[:num_words]


        positive_words = [word for word, result in sorted_words if result['compound_score'] > 0.0]
        negative_words = [word for word, result in sorted_words if result['compound_score'] < 0.0]
        
        return overall_compound_score, positive_words[:num_words], negative_words[-num_words:]

# Example usage:
# sia = SentimentAnalysis()
# input_string = "I love this product! It's amazing! The customer service is great."
# input_string = "According to the available information, he seems to be a researcher specialising in artificial intelligence (AI), who deals with the ethical implications of AI."
# overall_sentiment = sia.analyze_sentiment(input_string)
# overall_compound_score = overall_sentiment['compound_score']

# sorted_words = sia.sort_words_by_compound_distance(input_string, overall_compound_score)

# print("Overall Sentiment:")
# print(f"Sentiment: {overall_sentiment['sentiment']}")
# print(f"Compound Score: {overall_sentiment['compound_score']}")
# print(f"Positive Score: {overall_sentiment['positive_score']}")
# print(f"Negative Score: {overall_sentiment['negative_score']}")
# print(f"Neutral Score: {overall_sentiment['neutral_score']}")
# print("\n")

# print("Sorted Words by Compound Distance:")
# for word, result in sorted_words:
#     print(f"Word: {word}, Distance from Compound Score: {abs(result['compound_score'] - overall_compound_score)}")
#     print("\n")
    

# # def highlight_words(input_string, words_to_highlight):
# #     # Create a set of words to highlight for faster lookup
# #     words_to_highlight = set(words_to_highlight)
# #
# #     # Split the input string into words
# #     words = input_string.split()
# #
# #     # Create a list to hold the highlighted words
# #     highlighted_words = []
# #
# #     # Iterate over each word in the input string
# #     for word in words:
# #         # Check if the word is in the list of words to highlight
# #         if word.lower() in words_to_highlight:
# #             # If it is, wrap it in a span tag with a specific class
# #             highlighted_words.append(f'<span style="background-color: yellow;">{word}</span>')
# #         else:
# #             # If it is not, add the word as is
# #             highlighted_words.append(word)
# #
# #     # Join the highlighted words back into a single string
# #     highlighted_string = ' '.join(highlighted_words)
# #
# #     return highlighted_string

# def highlight_words(input_string, words_to_highlight):
#     # Create a set of words to highlight for faster lookup
#     # words_to_highlight = set(words.lower() for word in words_to_highlight)

#     # Tokenize the input string using NLTK's word tokenizer
#     words = word_tokenize(input_string)

#     # Create a list to hold the highlighted words
#     highlighted_words = []

#     # Iterate over each word in the input string
#     for word in words:
#         # Check if the word (in lowercase) is in the set of words to highlight
#         if word.lower() in words_to_highlight:
#             # If it is, wrap it in a span tag with an inline style
#             highlighted_words.append(f'<span style="background-color: yellow;">{word}</span>')
#         else:
#             # If it is not, add the word as is
#             highlighted_words.append(word)

#     # Join the highlighted words back into a single string
#     highlighted_string = ' '.join(highlighted_words)

#     return highlighted_string

# # Get only the words from sorted_words

# words_to_highlight = [word for word, result in sorted_words]
# # Reduce to 5
# words_to_highlight = words_to_highlight[:5]
# print(words_to_highlight)

# html = highlight_words(input_string, words_to_highlight[:5])

# print(html)