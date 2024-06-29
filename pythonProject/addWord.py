global custom_hate_speech_words
custom_hate_speech_words = []

def add_custom_hate_speech_words(word):
    custom_hate_speech_words.append(word);

def deleteWord(input):
    for word in custom_hate_speech_words:
        if word == input:
            custom_hate_speech_words.remove(input)
            return 1
    return 0