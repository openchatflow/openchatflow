from engine.chabot_data import MyChatbotData
from engine.chatbot_engine import get_pred, get_xs_ys, train, UNK
from actions.functions import get_actions

import json

DATA_PATH = "../data/data.json"


def init():
    training_data = json.load(open(DATA_PATH))
    chatbot_data = MyChatbotData(training_data)
    x,y = get_xs_ys(chatbot_data)
    nb_model = train(x,y)

    return chatbot_data, nb_model


def main():
    chatbot_data, nb_model = init()
    
    while True:
        try:
            query = input("User: ")
            pred = get_pred(query, nb_model, chatbot_data)
            answer = get_actions(pred, query, UNK)
            print("BOT:", answer)
        except KeyboardInterrupt:
            print("User: Exit!!")
            break

if __name__ == '__main__':
    main()