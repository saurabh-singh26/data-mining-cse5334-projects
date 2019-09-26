# Submitted by Saurabh Singh - 1001568347
import os, math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stopword = stopwords.words('english')
stemmer = PorterStemmer()
SMART_notation = "ltc.ltc" # Variant information for document and query vectors calculation
tokens = {} # This map stores the count matrix, and further updates to store normalized tf-idf score
idf = {} # This map store the idf scores for all the tokens in the document

# This method reads the file contents into a list 'doc', removes the lines parsed as new line character
# and return the list which is stored as a global variable
def readFile(filename):
    file = open(filename, "r", encoding='UTF-8')
    doc = file.readlines()
    file.close()
    for line in doc:
        if (line == "\n"):
            doc.remove(line)
    return doc

# This method parses over the global variable 'doc', and do the following over each individual line element:
# 1) Tokenize the line and store it in 'temp_tokens' variable
# 2) For each token stored in 'temp_tokens',if the token is not a stopword, stem it
# 3) Store the stemmed token in variable 'token_map' and increase the count if the same token is encountered again
# Variable 'token_map' is the count matrix for one line element in 'doc'.
# After we parse over all the line elements, the count matrix for document is stored in 'tokens' map
def preprocess_data():
    doc_id = 0 # This act as a key for tokens map identifying a particular document
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

# This method parses over all the tokens present in 'tokens' map, calculate its IDF, and store it in a global map 'idf'
def preprocess_idf():
    for doc_id in tokens:
        for token in tokens[doc_id]:
            if token not in idf:
                idf[token] = getidf(token)

# This method computes the normalized document tf-idf vector
def calculate_normalized_tf_idf():
    for doc_id in tokens:
        sum = 0
        max_tf = max(tokens[doc_id].values()) #Computed for 'augmented' tf variant
        # tf-tdf weight updated in the tokens map
        for token in tokens[doc_id]:
            tokens[doc_id][token] = gettf(tokens[doc_id][token], max_tf, SMART_notation[0]) * idf[token]
            sum = sum + math.pow(tokens[doc_id][token], 2)

        # Normalization
        for token in tokens[doc_id]:
            tokens[doc_id][token] = normalize(tokens[doc_id][token], sum, SMART_notation[2])

# This method calculates tf for a particular token depending on the tf variant
def gettf(tf, max_tf, variant='l'):
    if variant=='l':
        return (1 + math.log10(tf))
    if variant=='n':
        return tf
    if variant=='b':
        return 1 if tf>0 else 0
    if variant=='a':
        return 0.5 + ((0.5*tf)/max_tf)

# This method calculates idf for a particular token depending on the idf variant
def getidf(token, variant='t'):
    if token in idf:
        return idf[token] # Return value if already calculated
    N = len(doc)
    df_t = 0
    # Calculate df_t
    for doc_id in tokens:
        if token in tokens[doc_id]:
            df_t = df_t + 1
    # Depending on the variant return the corresponding idf value
    if variant=='t':
        if df_t==0:
            return -1
        return math.log10(N/df_t)
    if variant=='n':
        return 1
    if variant=='p':
        return max(0, math.log10((N-df_t)/df_t))

# This method normalize the tf-idf weights depending on the normalization variant used
def normalize(weight, sum, variant='c'):
    if variant=='c':
         return weight/math.sqrt(sum)
    if variant=='n':
        return weight

# This method computes the query vector depending on the SMART_notation values
def getqvec(string):
    # Computing count matrix
    string_token_map = {}
    for token in tokenizer.tokenize(string.lower()):
        if token not in stopword:
            stemmed_token = stemmer.stem(token)
            if stemmed_token in string_token_map:
                string_token_map[stemmed_token] = string_token_map[stemmed_token] + 1
            else:
                string_token_map[stemmed_token] = 1

    # Calculating tf-idf
    sum = 0
    for token in string_token_map:
        max_tf = max(string_token_map.values())
        tf_weight = gettf(string_token_map[token], max_tf, SMART_notation[4])
        if token in idf:
            string_token_map[token] =  tf_weight * idf[token]
        else:
            string_token_map[token] = tf_weight * math.log10(len(doc))
        sum = sum + math.pow(string_token_map[token], 2)

    # Normalization
    for token in string_token_map:
        string_token_map[token] = normalize(string_token_map[token], sum, SMART_notation[6])

    return string_token_map

# This method calculates the cosine similarity score and returns the document with maximum score, along with the score
def query(qstring):
    query_vector = getqvec(qstring) # Get the normalized query vector
    # Default score and paragraph values of there's no match with any of the documents
    score = 0
    paragraph = "No Match\n"
    for doc_id in tokens:
        doc_score = 0
        for token in query_vector:
            # If the query token does not exist in the document, its weight is 0 and thus score remains unchanged
            if token in tokens[doc_id]:
                doc_score = doc_score + query_vector[token] * tokens[doc_id][token]
        # Update the score and paragraph value if a new document is found with greater similarity score
        if doc_score > score:
            score = doc_score
            paragraph = doc[doc_id]
    return (paragraph, score)

# First read the file and store each paragraph as a document in 'doc' list
doc = readFile('./debate.txt')
# Then parse over the 'doc' list and compute count matrix
preprocess_data()
# Compute the idf value for each token based on the count matrix computed in above step
preprocess_idf()
# Compute the final normalized tf-idf score based on information from above 2 steps
calculate_normalized_tf_idf()