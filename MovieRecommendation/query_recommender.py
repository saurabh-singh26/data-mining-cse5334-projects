import pandas as pd
import numpy as np
import os, pickle, ast, operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from ast import literal_eval
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def get_recommendations(title):
    print("Entered the method...")
    cosine_sim2 = None
    metadata = None
    print("Starting of if else block")
    if os.path.isfile("cosineSimPickle.pkl"):
        print("Found cosineSimPickle")
        cosine_sim2 = pickle.load(open('cosineSimPickle.pkl', 'rb'))
        metadata = pickle.load(open('metadataPickle.pkl', 'rb'))
        # print(metadata)
    else:
        print("Did not found cosineSimPickle. Creating one...")
        # Load Movies Metadata,keywords,credits
        data_folder = 'data/'
        metadata = pd.read_csv(data_folder + 'movies_metadata.csv')
        credits = pd.read_csv(data_folder + 'credits.csv')
        keywords = pd.read_csv(data_folder + 'keywords.csv')

        metadata = metadata.drop([19730, 29503, 35587])

        # Convert IDs to int. Required for merging
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        metadata['id'] = metadata['id'].astype('int')

        # Merge keywords and credits into your main metadata dataframe
        metadata = metadata.merge(credits, on='id')
        metadata = metadata.merge(keywords, on='id')

        metadata = metadata.head(20000)
        # print(metadata)

        # Parse the stringified features into their corresponding python objects
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            metadata[feature] = metadata[feature].apply(literal_eval)

        # Define new director, cast, genres and keywords features that are in a suitable form.
        metadata['director'] = metadata['crew'].apply(get_director)

        features = ['cast', 'keywords', 'genres']
        for feature in features:
            metadata[feature] = metadata[feature].apply(get_list)
        # Print the new features of the first 3 films
        # print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))
        # print(metadata.head(3))

        # Apply clean_data function to your features.
        features = ['cast', 'keywords', 'director', 'genres']

        for feature in features:
            metadata[feature] = metadata[feature].apply(clean_data)

        # Create a new soup feature
        metadata['soup'] = metadata.apply(create_soup, axis=1)

        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(metadata['soup'])

        # Reset index of your main DataFrame and construct reverse mapping as before
        metadata = metadata.reset_index()


        cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
        print(type(cosine_sim2))
        print("Both data calculated. Dumping them...")
        pickle.dump(cosine_sim2, open('cosineSimPickle.pkl', 'wb+'))
        print("Dumped cosineSimPickle")
        pickle.dump(metadata, open('metadataPickle.pkl', 'wb+'))
        print("Dumped both")

    print("Ending of if else block")

    indices = pd.Series(metadata.index, index=metadata['title'])
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    result = []
    for entry in movie_indices:
        result.append(metadata['title'].iloc[entry])
    return result

# a = get_recommendations('The Dark Knight Rises')
# print (a)