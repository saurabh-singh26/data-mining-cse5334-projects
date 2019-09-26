import pandas as pd
import os, pickle, ast, operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def get_results(query):
    global prior_probability, post_probability
    initialize()
    if os.path.isfile("classifierPickle.pkl"):
        prior_probability = pickle.load(open('classifierPicklePrior.pkl', 'rb'))
        post_probability = pickle.load(open('classifierPicklePost.pkl', 'rb'))
    else:
        (prior_probability, post_probability) = build_and_save()
    return eval_result(query)

def eval_result(query):
    processed_query = pre_processing(query)
    genre_score = {}
    for genre in prior_probability.keys():
        score = prior_probability[genre]
        # print("For genre: ", genre, ", prior score: ", score)
        for token in processed_query:
            if (genre, token) in post_probability.keys():
                score = score * post_probability[(genre, token)]
                # print("token: ", token, ", score: ", score)
        genre_score[genre] = score
    sorted_score_map = sorted(genre_score.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score_map

def build_and_save():
    row_count = 0
    token_count = 0
    post_probability = {}
    token_genre_count_map = {}
    genre_count_map = {}
    for row in meta_data.itertuples():
        keywords = []
        genres = []
        for col in meta_cols.keys():
            col_values = getattr(row, col)
            parameters = meta_cols[col]
            # Paramter is None for tagline and overview columns, so appending data in keywords[]
            if parameters is None:
                keywords.append(col_values if isinstance(col_values, str) else "")
            # Else it is genres as it has a parameter "Name". So append in genres[]
            else:
                col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                for col_value in col_values:
                    for param in parameters:
                        genres.append(col_value[param])

        tokens = pre_processing(' '.join(keywords))
        for genre in genres:
            if genre in genre_count_map:
                genre_count_map[genre] += 1
            else:
                genre_count_map[genre] = 1
            for token in tokens:
                token_count += 1
                if (genre, token) in token_genre_count_map:
                    token_genre_count_map[(genre, token)] += 1
                else:
                    token_genre_count_map[(genre, token)] = 1

        row_count += 1
        # Uncomment below lines for reading specific number of rows from excel instead of the whole
        # if (row_count == 2):
        #     print(genre_count_map)
        #     break
    for (genre, token) in token_genre_count_map:
        post_probability[(genre, token)] = token_genre_count_map[(genre, token)] / token_count

    prior_probability = {x: genre_count_map[x]/row_count for x in genre_count_map}
    save(prior_probability, post_probability)
    return (prior_probability, post_probability)

def pre_processing(data_string):
    # for noise in noise_list:
    #     data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer.stem(t).lower())
    return processed_data

def save(prior_probability, post_probability):
    pickle.dump(prior_probability, open('classifierPicklePrior.pkl', 'wb+'))
    pickle.dump(post_probability, open('classifierPicklePost.pkl', 'wb+'))

def initialize():
    global meta_data, meta_cols, tokenizer, stopword, stemmer
    data_folder = "data/"
    meta_cols = {"genres":['name'], "overview":None, "tagline":None}
    meta_data = pd.read_csv(data_folder + 'movies_metadata.csv', usecols=meta_cols.keys())

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()

print(get_results(""))