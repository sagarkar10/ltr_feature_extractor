import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', 'textcat'])


def tokenize(text, remove_stopword=True, remove_punct=True, lowercase=False):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if remove_stopword:
            if token.is_stop:
                continue
        if remove_punct:
            if token.is_punct:
                continue
        if lowercase:
            tokens.append(token.text.lower())
        else:
            tokens.append(token.text)
    return tokens
