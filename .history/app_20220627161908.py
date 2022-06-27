from flask import Flask,render_template,request
from articleNews import Articles
from wtforms import Form, StringField,TextAreaField,validators
import pickle

app = Flask(__name__)

Articles = Articles()

# main
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


# @app.route('/classifier')
# def clasifier():
#     return render_template('classifier.html')

@app.route('/articles')
def articles():
    return render_template('articles.html',articles= Articles)

class inputForm(Form):
    newsHeadline = StringField('',[validators.Length(min=1,max=20000)])
    newsArticle = TextAreaField('',[validators.Length(min=5,max=20000000)])

@app.route('/sampleapi',methods=['POST'])
def sample_api ():
    Group_given = request.get_json()
    return {'status':'success','Group':Group_given}


# async with requestsasync.Session() as session:
#     response = await session.get('https://example.org')
#     print(response.status_code)
#     print(response.text)

@app.route('/classifiertry',methods=['GET','POST'])
def classifytry () :
    form = inputForm(request.form)
    if request.method == 'POST' and form.validate():
        newsArticleName = form.newsHeadline.data
        newsArticleText = form.newsArticle.data
        # print(newsArticleName)
        # print(newsArticleText)

        # API CALL
        PARAMS = {"DATA_ENTERED":"Profit loss % "}
        api_nlp = requests.post('https://predict-news-grp.herokuapp.com/getpred')
        print(api_nlp)
        # print(api_nlp.json())
        # group_nlp = api_nlp.json()
        # print(group_nlp['Group'])
        return render_template('classifier.html',form=form,HeadingValue=newsArticleName,ArticleValue=newsArticleText)
    return render_template('classifier.html',form=form)

@app.route('/classifier',methods=['GET','POST'])
def classify () :
    form = inputForm(request.form)
    if request.method == 'POST' and form.validate():
        newsArticleName = form.newsHeadline.data
        newsArticleText = form.newsArticle.data
        
        
        #Prediction of New Category
        tfidf_vectorizer = pickle.load(open("pickle/tfidf_vectorizer.pkl", 'rb'))
        nb_classifier = pickle.load(open("pickle/nb_classifier_for_tfidf_vectorizer.pkl", 'rb'))

        #Values encoded by LabelEncoder
        encoded = {0:'Business', 1:'Entertainment', 2:'Politics', 3:'Sports', 4:'Technology'}

        #Input
        # user_text = [input("Enter the news : ")]
        list_ARTICLE = []
        list_ARTICLE.append(newsArticleText)
        print(list_ARTICLE)

        #Transformation and Prediction of user text
        count = tfidf_vectorizer.transform(list_ARTICLE)
        prediction = nb_classifier.predict(count)

        print("\nNews Category : ", encoded[prediction[0]])
        groupValue = encoded[prediction[0]]
        return render_template('classifier.html',form=form,HeadingValue=newsArticleName,ArticleValue=newsArticleText,GroupValue=groupValue)
    return render_template('classifier.html',form=form)


if __name__ == '__main__':
    app.run(debug=True)