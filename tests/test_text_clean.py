from src.flaskproject.dao import *


def test_basic_text_clean():
    assert basic_text_cleaning(['3D print bureau', 'The Additive manufacturing company']) == \
           ["3d print bureau", "the additive manufacturing company"]


def test_basic_text_clean_neg():
    assert basic_text_cleaning([None]) == ['']
    assert basic_text_cleaning([None, 'NaN', 'nan']) == ['', 'nan', 'nan']


def test_url_clean():
    assert url_cleaning(['http://someLasercompany.com']) == ['someLasercompany']


def test_text_clean():
    text = 'Hello, this is new test case for cleaning.'
    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    assert text_clean(text, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False, wn=lem, ps=stem) == \
           ['hello', 'new', 'test', 'case', 'cleaning']
    assert text_clean(text, REGEX_ADV_CONST, do_lemmatize=False, do_stemming=True, wn=lem, ps=stem) == \
           ['hello', 'new', 'test', 'case', 'clean']


def test_perform_text_clean():
    text = ['Ceramic,Design,Design &amp; Simulation, Software Training', 'Materials Development,Polymer feedstock']
    cleaned_words, cleaned_sentence = perform_texts_clean(text, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False)
    assert cleaned_words == [['ceramic', 'design', 'design', 'amp', 'simulation', 'software', 'training'],
                             ['material', 'development', 'polymer', 'feedstock']]
    assert cleaned_sentence == ['ceramic design design amp simulation software training',
                                'material development polymer feedstock']
