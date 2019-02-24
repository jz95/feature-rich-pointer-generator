from nltk.corpus import brown
from nltk.corpus import treebank
from pickle import dump
from pickle import load
import nltk

def get_POS_tagged_sent(sentence):
    """
    Params:
    sentence should be a list of tokens, including punctuation.
    """

    #load the tagger 
    input = open('t4.pkl', 'rb')
    tagger = load(input)
    input.close()
    tagged_test_sentence = tagger.tag_sents([sentence])
    return tagged_test_sentence

def main():
    ret = get_POS_tagged_sent(['I', 'go', 'to', 'school', '.'])
    print(ret)

if __name__ == '__main__':
    main()
