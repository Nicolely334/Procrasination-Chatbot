import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem(word):
    """
    find the root form of the word
    e.g. "cries" -> "cry"
    """
    return stemmer.stem(word.lower())

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    that can be a word or punctuation character, or number
    e.g. "I feel so tired." -> ["I", "feel", "so", "tired", "."]
    """
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, all_words):
    """
    returns an array in which a 1 is placed for each known word that 
    exists in the sentence. if it is unknown, a 0 is placed instead
    e.g. 
    sentence = ["hello"]
    words = ["nicole", "hello", "is", "awesome"]
    bog = [0, 1, 0, 0]
    """
    #first, stem each word
    words_in_sentence = []
    for word in tokenized_sentence:
        stemmed_word = stem(word)
        words_in_sentence.append(stemmed_word)

    #initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for i, w in enumerate(all_words):
        if w in words_in_sentence: 
            bag[i] = 1

    return bag