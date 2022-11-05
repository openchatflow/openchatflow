import pandas as pd
import re
import random

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

class MyChatbotData:
    
    def __init__(self, json_obj):
        dfs = []
        for i, data in enumerate(json_obj["intents"]):
            # lowercase and remove punctuation
            intent = data["classname"]
            patterns = data["in_chat"].copy()
            answer = "##".join(data["response"].copy())
            for i, p in enumerate(patterns):
                p = p.lower()
                p = self.remove_punctuation(p)
                patterns[i] = p
            df = pd.DataFrame(list(zip([intent]*len(patterns), patterns, [answer]*len(patterns))), \
                              columns=['intent', 'phrase', 'answer'])
            dfs.append(df)
        self.df = pd.concat(dfs)
    
    def get_answer(self, intent):
        return random.choice(pd.unique(self.df[self.df['intent'] == intent]['answer'])[0].split("##"))
    
    def remove_punctuation(self, text):
        return punct_re_escape.sub('', text)
    
    def get_phrases(self, intent):
        return list(self.df[self.df['intent'] == intent]['phrase'])
    
    def get_intents(self):
        return list(pd.unique(self.df['intent']))
    
    def show_batch(self, size=5):
        return self.df.head(size)
    
    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    import json
    
    training_data = json.load(open('../../data/data.json','r'))
    chatbot_data = MyChatbotData(training_data)
    print(chatbot_data.get_intents())
    print(chatbot_data.get_phrases("greetings"))
    print(chatbot_data.get_answer("greetings"))