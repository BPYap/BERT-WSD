import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import wordnet as wn

WORDNET_POS = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}


def _get_info(lemma, pos, info_type):
    results = dict()

    wn_pos = WORDNET_POS[pos] if pos is not None else None
    morphemes = wn._morphy(lemma, pos=wn_pos) if pos is not None else []
    for i, synset in enumerate(set(wn.synsets(lemma, pos=wn_pos))):
        sense_key = None
        for l in synset.lemmas():
            if l.name().lower() == lemma.lower():
                sense_key = l.key()
                break
            elif l.name().lower() in morphemes:
                sense_key = l.key()
        assert sense_key is not None
        results[sense_key] = synset.examples() if info_type == 'examples' else synset.definition()

    return results


def get_glosses(lemma, pos):
    return _get_info(lemma, pos, info_type='gloss')


def get_example_sentences(lemma, pos):
    return _get_info(lemma, pos, info_type='examples')


def get_all_wordnet_lemma_names():
    results = []
    for pos, wn_pos in WORDNET_POS.items():
        results.append((pos, wn.all_lemma_names(pos=wn_pos)))

    return results
