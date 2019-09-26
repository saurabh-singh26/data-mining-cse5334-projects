import os, math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

filename = './debate.txt'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stopword = stopwords.words('english')
stemmer = PorterStemmer()
tokens = {}
idf = {}

def readFile():
    file = open(filename, "r", encoding='UTF-8')
    doc = file.readlines()
    file.close()
    for line in doc:
        if (line == "\n"):
            doc.remove(line)

    # for line in doc:
    #     print("@" + line + "@")
    # print(len(doc))
    return doc

def preprocess_data():
    doc_id = 0
    for line in doc:
        temp_tokens = tokenizer.tokenize(line)
        token_map = {}
        for t in temp_tokens:
            if t not in stopword:
                stemmed_token = stemmer.stem(t)
                if stemmed_token in token_map:
                    token_map[stemmed_token] = token_map[stemmed_token] + 1
                else:
                    token_map[stemmed_token] = 1
        tokens[doc_id] = token_map
        doc_id = doc_id + 1

def preprocess_idf():
    for doc_id in tokens:
        for token in tokens[doc_id]:
            if token not in idf:
                idf[token] = getidf(token)

def calculate_normalized_tf_idf():
    for doc_id in tokens:
        sum = 0
        for token in tokens[doc_id]:
            tokens[doc_id][token] = (1 + math.log10(tokens[doc_id][token])) * idf[token]
            sum = sum + math.pow(tokens[doc_id][token], 2)

        # Normalization
        for token in tokens[doc_id]:
            tokens[doc_id][token] = tokens[doc_id][token]/math.sqrt(sum)

def getidf(token):
    if token in idf:
        return idf[token]
    N = len(doc)
    df_t = 0
    for doc_id in tokens:
        if token in tokens[doc_id]:
            df_t = df_t + 1
    if df_t==0:
        return -1
    return math.log10(N/df_t)

def getqvec(string):
    string_lower_case = string.lower()
    string_token_tokenized = tokenizer.tokenize(string_lower_case)
    string_token_map = {}
    for token in string_token_tokenized:
        if token not in stopword:
            stemmed_token = stemmer.stem(token)
            if stemmed_token in string_token_map:
                string_token_map[stemmed_token] = string_token_map[stemmed_token] + 1
            else:
                string_token_map[stemmed_token] = 1

    # Calculating tf-idf
    sum = 0
    for token in string_token_map:
        if token in idf:
            string_token_map[token] = (1 + math.log10(string_token_map[token])) * idf[token]
        else:
            string_token_map[token] = (1 + math.log10(string_token_map[token])) * math.log10(len(doc))
        sum = sum + math.pow(string_token_map[token], 2)

    # Normalizing tf-idf
    for token in string_token_map:
        string_token_map[token] = string_token_map[token]/math.sqrt(sum)

    return string_token_map


def query(qstring):
    query_vector = getqvec(qstring)
    score = 0
    paragraph = "No Match\n"
    for doc_id in tokens:
        doc_score = 0
        for token in query_vector:
            if token in tokens[doc_id]:
                doc_score = doc_score + query_vector[token] * tokens[doc_id][token]
        if doc_score > score:
            score = doc_score
            paragraph = doc[doc_id]
    return (paragraph, score)

doc = readFile()
preprocess_data()
preprocess_idf()
calculate_normalized_tf_idf()

print("%.4f" % getidf(stemmer.stem("immigration")))
print("%.4f" % getidf(stemmer.stem("abortion")))
print("%.4f" % getidf(stemmer.stem("hispanic")))
print("%.4f" % getidf(stemmer.stem("the")))
print("%.4f" % getidf(stemmer.stem("tax")))
print("%.4f" % getidf(stemmer.stem("oil")))
print("%.4f" % getidf(stemmer.stem("beer")))

print(getqvec("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
print(getqvec("unlike any other time, it is under attack"))
print(getqvec("vector entropy"))
print(getqvec("clinton first amendment kavanagh"))

print("%s%.4f" % query("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
print("%s%.4f" % query("unlike any other time, it is under attack"))
print("%s%.4f" % query("vector entropy"))
print("%s%.4f" % query("clinton first amendment kavanagh"))
