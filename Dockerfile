FROM python:3.9

ADD . .

RUN apt update
RUN pip install numpy pandas spacy scikit-learn nltk telebot fuzzywuzzy pyTelegramBotAPI wikipedia python-dotenv
RUN python3 -m spacy download en_core_web_sm

CMD ["python", "src/telebot_main.py"]