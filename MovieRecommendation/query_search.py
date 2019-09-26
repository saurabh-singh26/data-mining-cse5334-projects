import pandas as pd
import ast, math, operator, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def create_inverted_index(x_data, x_cols):
    for row in x_data.itertuples():
        index = getattr(row, 'Index')
        data = []
        for col in x_cols.keys():
            if col != "id":
                col_values = getattr(row, col)
                parameters = x_cols[col]
                if parameters is None:
                     data.append(col_values if isinstance(col_values, str) else "")
                else:
                    col_values = ast.literal_eval(col_values if isinstance(col_values, str) else '[]')
                    for col_value in col_values:
                        for param in parameters:
                            data.append(col_value[param])
        insert(index, pre_processing(' '.join(data)))
        break

def pre_processing(data_string):
    for noise in noise_list:
        data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer.stem(t).lower())
    return processed_data

def insert(index, tokens):
    for token in tokens:
        if token in inverted_index:
            value = inverted_index[token]
            if index in value.keys():
                value[index] += 1
            else:
                value[index] = 1
                value["df"] += 1
        else:
            inverted_index[token] = {index: 1, "df": 1}

def build_doc_vector():
    for token_key in inverted_index:
        token_values = inverted_index[token_key]
        idf = math.log10(N / token_values["df"])
        for doc_key in token_values:
            if doc_key != "df":
                log_tf = 1 + math.log10(token_values[doc_key])
                tf_idf = log_tf * idf
                if doc_key not in document_vector:
                    document_vector[doc_key] = {token_key: tf_idf, "_sum_": math.pow(tf_idf, 2)}
                else:
                    document_vector[doc_key][token_key] = tf_idf
                    document_vector[doc_key]["_sum_"] += math.pow(tf_idf, 2)

    for doc in document_vector:
        tf_idf_vector = document_vector[doc]
        normalize = math.sqrt(tf_idf_vector["_sum_"])
        for tf_idf_key in tf_idf_vector:
            tf_idf_vector[tf_idf_key] /= normalize

def get_relevant_docs(query_list):
    relevant_docs = set()
    for query in query_list:
        if query in inverted_index:
            keys = inverted_index[query].keys()
            for key in keys:
                relevant_docs.add(key)
    if "df" in relevant_docs:
        relevant_docs.remove("df")
    return relevant_docs

def build_query_vector(processed_query):
    query_vector = {}
    sum = 0
    for token in processed_query:
        if token in inverted_index:
            tf_idf = (1 + math.log10(processed_query.count(token))) * math.log10(N/inverted_index[token]["df"])
            query_vector[token] = tf_idf
            sum += math.pow(tf_idf, 2)
    sum = math.sqrt(sum)
    for token in query_vector:
        query_vector[token] /= sum
    return query_vector

def cosine_similarity(relevant_docs, query_vector):
    score_map = {}
    for doc in relevant_docs:
        score = 0
        for token in query_vector:
            score += query_vector[token] * (document_vector[doc][token] if token in document_vector[doc] else 0)
        score_map[doc] = score
    sorted_score_map = sorted(score_map.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score_map[:10]

def get_results(query):
    global inverted_index, document_vector
    initialize()
    if os.path.isfile("invertedIndexPickle.pkl"):
        inverted_index = pickle.load(open('invertedIndexPickle.pkl', 'rb'))
        document_vector = pickle.load(open('documentVectorPickle.pkl', 'rb'))
    else:
        print("In else of get_scores:")
        build()
        save()
    return eval_score(query)

def initialize():
    global data_folder, credits_cols, meta_cols, noise_list, credits_data, meta_data, N, tokenizer, stopword, stemmer, inverted_index, document_vector

    # Data configurations
    data_folder = "data/"
    credits_cols = {"id": None, "cast":['character', 'name'], "crew":['name']}
    meta_cols = {"id": None, "genres":['name'], "original_title":None, "overview":None,
                     "production_companies":['name'], "tagline":None, "poster_path":None}
    noise_list = ['(voice)', '(uncredited)']

    # Read data
    credits_data = pd.read_csv(data_folder + 'credits.csv', usecols=credits_cols.keys(), index_col="id")
    meta_data = pd.read_csv(data_folder + 'movies_metadata.csv', usecols=meta_cols.keys(), index_col="id")
    # Total number of documents = number of rows in movies_metadata.csv
    N = meta_data.shape[0]

    # Pre-processing initialization
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()

    inverted_index = {}
    document_vector = {}
    print("Initialized")

def build():
    print("Creating inverted index for credits data...")
    create_inverted_index(credits_data, credits_cols)
    print("Creating inverted index for meta data...")
    create_inverted_index(meta_data, meta_cols)
    print("Building doc vector...")
    build_doc_vector()
    print("Built index and doc vector")

def save():
    pickle.dump(inverted_index, open('invertedIndexPickle.pkl', 'wb+'))
    pickle.dump(document_vector, open('documentVectorPickle.pkl', 'wb+'))
    print("Saved both")

def eval_score(query):
    processed_query = pre_processing(query)
    # print(processed_query)
    relevant_docs = get_relevant_docs(processed_query)
    # print(relevant_docs)
    query_vector = build_query_vector(processed_query)
    # print(query_vector)
    sorted_score_list = cosine_similarity(relevant_docs, query_vector)
    search_result = get_movie_info(sorted_score_list)
    # print(search_result)
    return search_result

def get_movie_info(sorted_score_list):
    result = []
    for entry in sorted_score_list:
        doc_id = entry[0]
        row = meta_data.loc[doc_id]
        info = (row["original_title"],
                row["overview"] if isinstance(row["overview"], str) else "",
                entry[1],
                row["poster_path"])
        result.append(info)
    return result

# query = "Taisto Olavi Kasurinen"





