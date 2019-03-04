# -*- coding: utf-8 -*-
from pickle import dump, load

def get_POS_tagged_sent(sentence):
    """
    Params:
    sentence should be a list of tokens, including punctuation.
    """

    #load the tagger 
    with open('/Users/j.zhou/mlp_project/pointer-generator/pos_tagger/t4.pkl', 'rb') as f:
        tagger = load(f)
    tagged_test_sentence = tagger.tag_sents([sentence])
    return tagged_test_sentence

def main():
    ret = get_POS_tagged_sent(['I', 'go', 'to', 'school', '.'])
    print(ret)

if __name__ == '__main__':
    main()
