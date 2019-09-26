from flask import Flask, render_template, request, jsonify

application = Flask(__name__)
application.debug = True


@application.route('/')
def hello_world():
    return render_template('index.html')


@application.route('/search/', methods=['GET', 'POST'])
def search():
    search_query = request.form.get('query')
    print(search_query)
    import query_search
    data = {'results': query_search.get_results(search_query)}
    data = jsonify(data)
    return data


@application.route('/classify/', methods=['GET', 'POST'])
def classify():
    classify_query = request.form.get('classify')
    import query_classifier
    data = {'results': query_classifier.get_results(classify_query)}
    data = jsonify(data)
    return data


@application.route('/recommend/', methods=['GET', 'POST'])
def recommend():
    recommend_query = request.form.get('recommend')
    print("Query is: ", recommend_query)
    import query_recommender
    data = {'results': query_recommender.get_recommendations(recommend_query)}
    data = jsonify(data)
    print(data)
    return data


if __name__ == '__main__':
    application.run()