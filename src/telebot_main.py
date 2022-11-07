from engine.chabot_data import MyChatbotData
from engine.chatbot_engine import get_pred, get_xs_ys, train, remove_punctuation, UNK
from actions.basic_functions import get_actions

import json
import os
import telebot

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH","data/data.json")
API_KEY = os.getenv("API_KEY")
bot = telebot.TeleBot(API_KEY)


def init():
    training_data = json.load(open(DATA_PATH))
    chatbot_data = MyChatbotData(training_data)
    x,y = get_xs_ys(chatbot_data)
    nb_model = train(x,y)

    return chatbot_data, nb_model

chatbot_data, nb_model = init()


@bot.message_handler(content_types=['text'])
def message_func(message):
    query = message.text
    pred = get_pred(query, nb_model, chatbot_data)
    query = remove_punctuation(query)
    answer = get_actions(pred, query, UNK)
    bot.send_message(message.chat.id, answer)

bot.polling()