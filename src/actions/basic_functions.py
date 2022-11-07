import datetime
import wikipedia


def date_time():
    date = str(datetime.datetime.now()).split(' ')
    return f"current date is {date[0]} and current time is {date[1]}"


def what_is_cls(query):
    try:
        result = wikipedia.summary(query,5) + "\n"+ wikipedia.page(query).url
    except:
        result = "I'm sorry, I don't know, may be you can spell it correctly and don't use abbreviation"
    
    return result


def get_actions(pred, query, UNK):
    if "act" in pred:
        pred = pred.split('.')[-1]
        if pred == "date_time":
            answer = date_time()
        elif pred == "what_is_cls":
            answer = what_is_cls(query)
    elif pred == UNK:
        answer = what_is_cls(query)
    else:
        answer = pred
    
    return answer

if __name__ == '__main__':
    print(date_time())
    print(what_is_cls("Yann LeCun"))