import re
import unicodedata
from fuzzywuzzy import process
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_sm',disable=['parser','tagger'])

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

UNK = "I'm sorry, I don't know, may be you can spell it correctly and don't use abbreviation"


EMOJIS = [[':)', '😀'],[';)', '😉'],[':(', '😞'],[';((', '😢'],[':p', '😛']]
_emoji_re = '[\U00010000-\U0010ffff]+'
emoji_re = re.compile(_emoji_re, flags=re.UNICODE)


def tokenize_nd_join(text):
    doc = nlp(text.lower())
    return " ".join(tok.text for tok in doc if tok.text.strip() not in stopwords)


def get_xs_ys(chatbot_data):
    x, y = [], []
    intents = chatbot_data.get_intents()
    for i in intents:
        phrases = chatbot_data.get_phrases(i)
        x += [tokenize_nd_join(phrase) for phrase in phrases]
        y += [i]*len(phrases)
    return x, y


def train(x,y):
    vect = CountVectorizer(ngram_range=(1,2),max_features=None)
    nb = Pipeline([('vect',vect),('clf',ComplementNB(alpha=1.0,norm=False))])
    nb.fit(x,y)
    return nb


def nb_pred_top3(query, nb_model, chatbot_data):
    tokenized_query = tokenize_nd_join(query)
    pred_prob = nb_model.predict_proba([tokenized_query])
    preds_sorted = np.argsort(pred_prob)
    top3 = preds_sorted[:,-1],preds_sorted[:,-2],preds_sorted[:,-2]
    if pred_prob[0,top3[0]] > (pred_prob[0,top3[1]] + pred_prob[0,top3[2]]):
        pred = nb_model.named_steps['clf'].classes_[top3[0]][0]
        return chatbot_data.get_answer(pred)
    return UNK


def remove_punctuation(text):
    text = punct_re_escape.sub('', text)
    return text


def ascii_normalize(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode("utf-8") 


def emoji_normalize(text):
    for e1, e2 in EMOJIS:
        text = text.replace(e1, e2)
    return text


def is_emoji(text):
    emoji = "".join(re.findall(_emoji_re, text))
    return emoji == text


def emoji_isolate(text):
    EMJ = "__EMOJI__"
    emoji_list = re.findall(_emoji_re, text)
    text = emoji_re.sub(f" {EMJ} ", text)
    new_str, ctr = [], 0
    for tok in text.split():
        if tok == EMJ:
            new_str.append(emoji_list[ctr])
            ctr += 1
        else:
            new_str.append(tok)
    return " ".join(new_str).strip()


def preprocess(text):
    text = text.lower()
    text = ascii_normalize(text) or text
    text = emoji_normalize(text) or text
    text = emoji_isolate(text) or text
    text = remove_punctuation(text) or text
    return text


def exact_match(query, chatbot_data):
    intents = chatbot_data.get_intents()
    for i in intents:
        phrases = chatbot_data.get_phrases(i)
        if query in phrases:
            return chatbot_data.get_answer(i)
    return UNK


def fuzzy_matching(query, chatbot_data):
    intents = chatbot_data.get_intents()
    for i in intents:
        phrases = chatbot_data.get_phrases(i)
        match, score = process.extractOne(query, phrases)
        if score > 90:
            return chatbot_data.get_answer(i)
    return UNK


def get_pred(query, nb_model, chatbot_data):
    query = query.lower()
    pred = exact_match(query, chatbot_data)
    if pred == UNK: pred = exact_match(preprocess(query), chatbot_data)
    if pred == UNK: pred = fuzzy_matching(query, chatbot_data)
    if pred == UNK: pred = fuzzy_matching(preprocess(query), chatbot_data)
    if pred == UNK: pred = nb_pred_top3(query, nb_model, chatbot_data)
    if pred == UNK: pred = nb_pred_top3(preprocess(query), nb_model, chatbot_data)
    return pred


if __name__ == '__main__':
    from chabot_data import MyChatbotData
    import json

    training_data = json.load(open('../../data/data.json','r'))
    chatbot_data = MyChatbotData(training_data)

    x, y = get_xs_ys(chatbot_data)
    nb_model = train(x, y)
    # print(exact_match(preprocess("Hi"), chatbot_data))
    # print(exact_match(preprocess("uwuwuwuw"), chatbot_data))
    # print(exact_match(preprocess("who are you"), chatbot_data))

    # print(fuzzy_matching(preprocess("Hallo"), chatbot_data))
    # print(fuzzy_matching(preprocess("uwuwuwuw"), chatbot_data))
    # print(fuzzy_matching(preprocess("who are you"), chatbot_data))

    print(get_pred("Hallo",nb_model, chatbot_data))
    print(get_pred("uwuwuwuw",nb_model, chatbot_data))
    print(get_pred("who are you",nb_model, chatbot_data))
    print(get_pred("what time is it",nb_model, chatbot_data))
    print(get_pred("what is loss function",nb_model, chatbot_data))