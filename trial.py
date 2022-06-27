import requests


async def fun():
    PARAMS = {"DATA_ENTERED":"Profit loss % "}
    api_nlp = await requests.post('https://predict-news-grp.herokuapp.com/getpred',data=PARAMS)
    print(api_nlp)


print(fun())