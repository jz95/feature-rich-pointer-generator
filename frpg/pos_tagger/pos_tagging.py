# -*- coding: utf-8 -*-
from pickle import load
import pkg_resources

TAGGER_PATH = pkg_resources.resource_filename(__name__, 'POS_TAGGER.pkl')

with open(TAGGER_PATH, 'rb') as f:
    TAGGER = load(f)


def get_POS_tagged_sent(sentence):
    """
    Params:
    sentence should be a list of tokens, including punctuation.
    """
    return TAGGER.tag_sents([sentence])[0]


def main():
    ret = get_POS_tagged_sent(['I', 'go', 'to', 'school', '.'])
    print(ret)

if __name__ == '__main__':
    main()