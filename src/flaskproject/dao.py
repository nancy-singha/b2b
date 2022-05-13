from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
from numpy import savetxt
from geopy.geocoders import Nominatim
from geopy import geocoders as gc
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import geopy.distance as gpd
import pandas as pd
import numpy as np
import re
import nltk
import string
import random
import warnings

warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

englishStopwords = stopwords.words('english')
tf_idf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=100)
REGEX_ADV_CONST = "[^-:,0-9A-Za-z ]"
REGEX_BASIC_CONST = "[^-,0-9A-Za-z ]"
SPACE = " "
BLANK = ""
random.seed(42)


def basic_text_cleaning(texts):
    """
    Apply basic cleaning like converting text to lowercase.
    :param texts: list of strings to be cleaned
    :return: list of cleaned text as sentence
    """
    cleaned_sentence_text = []
    for txt in texts:
        if pd.isna(txt) or pd.isnull(txt):
            cleaned_sentence_text.append(BLANK)
        else:
            clean_text = txt.lower()
            cleaned_sentence_text.append(clean_text)
    return cleaned_sentence_text


def text_clean(txt, regexp, do_lemmatize, do_stemming, wn: WordNetLemmatizer, ps: PorterStemmer):
    """
    Clean the provided string applying the regular expression.
    :param txt: string to be cleaned
    :param regexp: regular expression to apply
    :param do_lemmatize: flag to indicate if lemmatization has to be applied
    :param do_stemming: flag to indicate if stemming has to be applied
    :param wn: instance of WordNetLemmatizer
    :param ps: instance of PorterStemmer
    :return: cleaned text
    """
    if pd.isna(txt) or pd.isnull(txt):
        return BLANK
    clean_text = re.sub(regexp, BLANK, txt)
    clean_text = clean_text.lower()
    clean_text = clean_text.split(sep=',')
    clean_text = [word_tokenize(words) for words in clean_text]
    clean_text = [word for words in clean_text for word in words]
    if do_stemming:
        clean_text = [ps.stem(word) for word in clean_text
                      if word not in set(englishStopwords)]
    if do_lemmatize:
        clean_text = [wn.lemmatize(word) for word in clean_text
                      if word not in set(englishStopwords)]
    return clean_text


def perform_texts_clean(texts, regexp, do_lemmatize, do_stemming):
    """
    Clean the provided strings applying the regular expression.
    :param texts: list of strings to be cleaned
    :param regexp: regular expression to apply
    :param do_lemmatize: flag to indicate if lemmatization has to be applied
    :param do_stemming: flag to indicate if stemming has to be applied
    :return: list of list containing words that are cleaned and list of cleaned text as sentence
    """
    ps = PorterStemmer()
    wn = WordNetLemmatizer()
    cleaned_words_per_text = []
    cleaned_sentence_text = []
    for txt in texts:
        if pd.isna(txt) or pd.isnull(txt):
            cleaned_words_per_text.append([])
            cleaned_sentence_text.append(BLANK)
        else:
            doc = txt
            # Apply regular expression
            clean_text = text_clean(doc, regexp, do_lemmatize, do_stemming, wn, ps)
            cleaned_words_per_text.append(clean_text)
            cleaned_sentence_text.append(SPACE.join(clean_text))

    return cleaned_words_per_text, cleaned_sentence_text


def url_cleaning(links):
    """
    Clean the url.
    :param links: list of url strings
    :return: list of cleaned url strings
    """
    cleaned_texts = []
    for link in links:
        if pd.isna(link) or pd.isnull(link):
            cleaned_texts.append('')
        else:
            hostname = link
            hostname = hostname.removeprefix('https://')
            hostname = hostname.removeprefix('http://')
            hostname = hostname.removeprefix('www.')
            hostname = hostname[0:find_index(hostname):]
            cleaned_texts.append(hostname)
    return cleaned_texts


def find_index(link: string):
    """
    Find the start index of specific strings in the input string.
    :param link: url in which a sub-string is to be found.
    :return: valid index if the sub string is found
    """
    idx = link.rfind('.com')
    if idx == -1:
        idx = link.rfind('.co.uk')
    if idx == -1:
        idx = link.rfind('.ac.uk')
    if idx == -1:
        idx = link.rfind('.ca')
    if idx == -1:
        idx = link.rfind('.org')
    if idx == -1:
        idx = link.rfind('.eu')
    if idx == -1:
        idx = link.rfind('.de')

    return idx


def convert_to_numeric_tsv(dff: DataFrame, filename):
    """
    Save the DataFrame contents to a TSV file.
    :param dff: pandas DataFrame to be put into the file
    :param filename: name of the file to be created
    :return: created file
    """
    np_arr = dff.to_numpy(dtype=np.float64)
    savetxt(filename, np_arr, delimiter='\t')


def convert_to_str_tsv(dff: DataFrame, filename):
    """
    Save the DataFrame contents to a TSV file.
    :param dff: pandas DataFrame to be put into the file
    :param filename: name of the file to be created
    :return: created file
    """
    dffs = dff.astype(str)
    np_arr = dffs.to_numpy()
    savetxt(filename, np_arr, delimiter='\t', encoding='utf-8', fmt='%s')


def convert_to_bins(finance_info):
    """
    Convert the numerical data to categories.
    :return: category corresponding to financial value
    """
    q = 5
    ranked_finance = finance_info.rank(method='first')
    categories = ['very-low', 'low', 'medium', 'high', 'very-high']
    bins = pd.qcut(ranked_finance, q, labels=categories)

    return bins


def get_cleaned_search_phrase(query):
    """
    Cleans the search query so that it can be fed to the embedding model.
    :param query: search text
    :return: cleaned search text
    """
    # different search parameters to be concatenated with space before setting to the below variable
    txt_cln_search_phrase = text_clean(query, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False,
                                       wn=WordNetLemmatizer(), ps=PorterStemmer())
    cleaned_search_phrase = ' '.join(txt_cln_search_phrase)
    return cleaned_search_phrase


def compute_ranking_fin_accr(scores, accreditation, revenue):
    """
    Compute the scores with respect to accreditation and financial information.
    :param scores: all scores
    :param accreditation: accreditation details
    :param revenue: finance details
    :return: updated scores
    """
    rank_accr_base = adjust_score_accr(scores, accreditation)
    rank_fin_base = adjust_score_finance(rank_accr_base, revenue)
    return rank_fin_base


def get_similarity_score(vocab_vector, search_vector, is_fin_accr_ranking_reqd, clean_df):
    """
    Calculates cosine similarity score between input and search vectors.
    :param vocab_vector: vocabulary where the query is to be searched
    :param search_vector: search text vector
    :param is_fin_accr_ranking_reqd: flag indicating if other factors should be considered for ranking
    :param clean_df: pandas data frame containing cleaned text
    :return: final scores
    """
    sim_score = get_cosine_similarity(vocab_vector, search_vector)
    if not is_fin_accr_ranking_reqd:
        return sim_score

    scores_copy = sim_score.copy()
    final_scores = compute_ranking_fin_accr(scores_copy, clean_df.Accreditation, clean_df.Revenue)

    return final_scores


def get_top_k(k, scores, clean_df):
    """
    Fetches the top scores.
    :param k: number of top results required
    :param scores: collection of all scores
    :param clean_df: pandas dataframe containing cleaned text
    :return: the top k scores in descending order
    """
    comp_score_index_data = []
    ind = np.argpartition(scores, -k, axis=0)[-k:]
    indices = ind.reshape(-1).tolist()
    desc_index_scores = {}
    for i in reversed(indices):
        desc_index_scores[i] = scores[i]
    sorted_desc_index_scores = {k: v for k, v in
                                sorted(desc_index_scores.items(), key=lambda item: item[1], reverse=True)}
    for index, score in sorted_desc_index_scores.items():
        comp_score_index_data.append((clean_df.Comp[index], score, index))
        print("Company {0} with similarity score {1}".format(clean_df.Comp[index], scores[index]))
    return comp_score_index_data


def get_top_k_scores(vocab_vector, search_vector, k, is_fin_accr_ranking_reqd, clean_df):
    """
    Calculates the top scores for the specified query.
    :param vocab_vector: vocabulary where the query is to be searched
    :param search_vector: search text vector
    :param k: number of top results required
    :param is_fin_accr_ranking_reqd: flag indicating if other factors should be considered for ranking
    :param clean_df: pandas dataframe containing cleaned text
    :return: list of tuples containing company name, score and index of the supplier
    """
    scores = get_similarity_score(vocab_vector, search_vector, is_fin_accr_ranking_reqd, clean_df)
    score_data = get_top_k(k, scores.flatten(), clean_df)
    return score_data


def run_tfidf_col(col_name, query, clean_df):
    """
    Run TF-IDF model for a specified column
    :param col_name: name of the column
    :param query: search text
    :param clean_df: pandas dataframe containing cleaned text
    :return: similarity score between the input and query
    """
    tfidf_col_model = TfidfVectorizer(analyzer='word', stop_words='english', max_features=100)
    col_data_vectors = fit_col_vector(tfidf_col_model, col_name, clean_df)
    query_data_vector = predict_search_col_vector(tfidf_col_model, query)
    return get_similarity_score(col_data_vectors, query_data_vector, False, clean_df)


def fit_col_vector(tfidf_col_model: TfidfVectorizer, col_name, clean_df):
    """
    Transforms the input data to vector form thereby learning the words.
    :param tfidf_col_model: TF-IDF model instance
    :param col_name: name of the column which serves as the input on which the model is to be trained
    :param clean_df: pandas dataframe containing cleaned text
    :return: pandas Dataframe containing vectors for words in the column
    """
    # column data is already cleaned and combined beforehand
    column = clean_df[col_name]
    column_data = column.values.tolist()
    # instantiate the vectorizer object
    # convert the documents into a matrix
    tfidf_wm = tfidf_col_model.fit_transform(column_data)  # retrieve the terms found in the corpora
    df_tfidf_vect = pd.DataFrame(data=tfidf_wm.toarray(), index=clean_df.Comp)
    return df_tfidf_vect


def predict_search_col_vector(tfidf_col_model, query):
    """
    Transforms search query into vector form based on the words learnt.
    :param tfidf_col_model: TF-IDF model instance
    :param query: search text
    :return: pandas DataFrame containing the vector for the search text
    """
    output = tfidf_col_model.transform([query])
    np_out_arr = output.todense()
    dfo = pd.DataFrame(data=np_out_arr, columns=tfidf_col_model.get_feature_names())
    return dfo


def fit_vectorizer(clean_df):
    """
    Transforms the input data to vector form thereby learning the words.
    :param clean_df: pandas dataframe containing cleaned text
    :return: pandas Dataframe containing vectors for words in the dataset
    """
    sentence = clean_df.apply(' '.join, axis=1)
    sentences = sentence.values.tolist()
    tfidf_wm = tf_idf_vectorizer.fit_transform(sentences)  # retrieve the terms found in the corpora
    tfidf_tokens = tf_idf_vectorizer.get_feature_names()
    df_tfidf_vect = pd.DataFrame(data=tfidf_wm.toarray(), index=clean_df.Comp, columns=tfidf_tokens)
    return df_tfidf_vect


def predict_search_vector(query):
    """
    Transforms search query into vector form based on the words learnt.
    :param query: search text
    :return: pandas DataFrame containing the vector for the search text
    """
    cleaned_search_phrase = get_cleaned_search_phrase(query)
    output = tf_idf_vectorizer.transform([cleaned_search_phrase])
    np_out_arr = output.todense()
    dfo = pd.DataFrame(data=np_out_arr, columns=tf_idf_vectorizer.get_feature_names())
    return dfo


def get_cosine_similarity(input_data, search):
    """
    Compute cosine similarity scores.
    :param input_data: supplier data
    :param search: query to search
    :return: cosine similarity scores
    """
    similarity = cosine_similarity(input_data, search)
    return similarity


def convert_to_tsv(fitted_data, search_data, query, filename1, filename2, clean_df):
    """
    Save the values to a file in tsv format.
    :param fitted_data: supplier data as sentence
    :param search_data: company name
    :param query: search query
    :param filename1: name of the file to save supplier data embeddings
    :param filename2: name of the file to save company name
    :param clean_df: pandas dataframe containing cleaned text
    :return: None
    """
    dff = pd.concat([fitted_data, search_data], ignore_index=True)
    convert_to_numeric_tsv(dff, filename1)

    comp_name = pd.DataFrame(clean_df.Comp)
    cleaned_search_phrase = get_cleaned_search_phrase(query)
    search_ph = pd.DataFrame([cleaned_search_phrase])
    new_df = pd.concat([comp_name, search_ph])
    convert_to_str_tsv(new_df, filename2)


def valid_accr_indices(accreditation):
    """
    The index of the companies that have accreditation.
    :param accreditation: accreditation details
    :return: dictionary with key as index and the number of certifications as value
    """
    ac_list = accreditation
    indices_length = {}

    for i in range(len(ac_list)):
        txt = accreditation[i]
        if not (pd.isna(txt) or pd.isnull(txt)):
            items = txt.split(sep=',')
            indices_length[i] = len(items)

    return indices_length


def calculate_distance(location_phrase, clean_df):
    """
    Calculate the distance between the specified location and the locations in the dataset.
    :param location_phrase: the location user is looking for
    :param clean_df: pandas dataframe containing cleaned text
    :return: distances between the specified location and the locations in the dataset
    """
    app_name = "my-app"
    # calling the Nominatim tool
    n = Nominatim(user_agent=app_name)
    gc_p = gc.photon
    p = gc_p.Photon(user_agent=app_name)

    search_lat = 0.0
    search_lng = 0.0
    loc = n.geocode(location_phrase)

    if loc is None:
        loc = p.geocode(location_phrase)

    if loc is not None:
        search_lat = loc.latitude
        search_lng = loc.longitude

    distances = []
    search_coord = (search_lat, search_lng)

    for i in range(len(clean_df.Comp)):
        latitude = clean_df.latitudes[i]
        longitude = clean_df.longitudes[i]

        if pd.isna(latitude) or pd.isnull(latitude):
            latitude = 0.0
        if pd.isna(longitude) or pd.isnull(longitude):
            longitude = 0.0

        existing_coord = (latitude, longitude)
        distance = gpd.distance(search_coord, existing_coord).kilometers
        distances.append(distance)

    return distances


def get_category_score(is_numpy):
    """
    Get the value to add or subtract, for corresponding financial state of a company.
    :param is_numpy: flag specifying if the scores are numpy array or not
    :return: the value to be added or subtracted
    """
    if is_numpy:
        return {'very-low': -0.2, 'low': -0.1, 'medium': 0.0, 'high': 0.1, 'very-high': 0.2}
    return {'very-low': -2.0, 'low': -1.0, 'medium': 0.0, 'high': 1.0, 'very-high': 2.0}


def get_accr_score(is_numpy):
    """
    Get the value to add, if accreditation is present.
    :param is_numpy: flag specifying if the scores are numpy array or not
    :return: the value to be added
    """
    if is_numpy:
        return 0.1
    return 1.0


def get_max_certification_value(num_certifications):
    """
    Get maximum certifications allowed.
    :param num_certifications: actual number of certifications
    :return: max number of certifications allowed if more else the same value
    """
    if num_certifications > 2:
        return 2
    return num_certifications


def adjust_score_accr(scores, clean_df):
    """
    Update the scores according to the accreditation of the company.
    :param scores: scores to be modified
    :param clean_df: pandas dataframe containing cleaned text
    :return: updated scores
    """
    indices_n_length = valid_accr_indices(clean_df)
    is_numpy = isinstance(scores, np.ndarray)

    if not is_numpy:
        adjusted_scores = scores.clone()
    else:
        adjusted_scores = scores.copy()

    for k, v in indices_n_length.items():
        step = get_accr_score(is_numpy)
        enhance_score = step * get_max_certification_value(v)
        score = adjusted_scores[k]
        enhanced_score = score + enhance_score
        adjusted_scores[k] = enhanced_score

    return adjusted_scores


def adjust_score_finance(scores, revenue):
    """
    Update the scores according to the financial state of the company.
    :param scores: scores to be modified
    :param revenue: finance details
    :return: updated scores
    """
    fin_bins = revenue
    is_numpy = isinstance(scores, np.ndarray)

    if not is_numpy:
        adjusted_scores = scores.clone()
    else:
        adjusted_scores = scores.copy()

    cat_score = get_category_score(is_numpy)

    for i in range(len(fin_bins)):
        category = fin_bins[i]
        enhance_score = cat_score[category]
        enhanced_score = adjusted_scores[i] + enhance_score
        adjusted_scores[i] = enhanced_score

    return adjusted_scores


def use_sentence_embedder(corpus, query, clean_df, k=5):
    """
    Get the top k scores for the data.
    :param corpus: the supplier data as sentences
    :param query: the search query as sentence
    :param clean_df: cleaned data
    :param k: number of top results required
    :return: top k ranking scores and their index
    """
    embedder = SentenceTransformer('multi-qa-distilbert-cos-v1')
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, convert_to_numpy=False)
    query_embedding = embedder.encode(query, convert_to_tensor=True, convert_to_numpy=False)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    scores_copy = cos_scores.clone()
    final_scores = compute_ranking_fin_accr(scores_copy, clean_df.Accreditation, clean_df.Revenue)

    top_k = min(k, len(corpus))
    top_results = torch.topk(final_scores, k=top_k)

    return top_results


def run_fuzzy_col(query, col_search_data):
    """
    Run fuzzy logic for the provided data.
    :param query: search query
    :param col_search_data: column data to be matched against
    :return: match scores
    """
    processed_scores = []

    for c in col_search_data:
        score = fuzz.UWRatio(query, c)
        processed_scores.append(score / 100)

    return processed_scores


def normalize_values(values):
    """
    Normalize values to a scale of 0 to 1.
    :param values: values to be normalized
    :return: numpy array containing normalized values
    """
    normalized_values = (np.max(values) - values) / (np.max(values) - np.min(values))
    return normalized_values


def get_company_scores(company_name, companies):
    """
    Get similarity scores for companies.
    :param company_name: name of the company to search
    :param companies: list of all companies
    :return: company name similarity scores
    """
    if company_name is None:
        return None

    cleaned_company_name = get_cleaned_search_phrase(company_name)
    company_scores = run_fuzzy_col(cleaned_company_name, companies)
    return company_scores


def get_weighted_average(scores: pd.DataFrame, weights: np.ndarray, clean_df):
    """
    Calculate weighted average score. It considers financial information and accreditation to get the final scores.
    :param scores: pandas Dataframe containing the scores for different search parameters
    :param weights: numpy array containing the weights to be applied to calculate the average
    :param clean_df: pandas dataframe containing cleaned text
    :return: final ranking scores
    """
    if len(scores) == 0:
        return 0

    scores_np = scores.to_numpy()
    weighted_avg = np.average(scores_np, weights=weights, axis=1)
    final_scores = compute_ranking_fin_accr(weighted_avg, clean_df.Accreditation, clean_df.Revenue)

    return final_scores


def compute_company_score(companies, clean_df):
    """
    Compute the company name scores.
    :param companies: company name to be searched
    :param clean_df: pandas dataframe containing cleaned text
    :return: numpy array of company name similarity scores
    """
    company_scores = get_company_scores(companies, clean_df.Comp)
    final_scores = np.asarray(company_scores)
    return final_scores


def compute_capability_score(capability, clean_df):
    """
    Compute the capability scores using fuzzy logic.
    :param capability: capability to be searched
    :param clean_df: pandas dataframe containing cleaned text
    :return: numpy array of capability similarity scores
    """
    cleaned_capability = get_cleaned_search_phrase(capability)
    capability_scores = run_fuzzy_col(cleaned_capability, clean_df.Capabilities)
    capability_scores_np = np.asarray(capability_scores)
    return capability_scores_np


def compute_capability_tfidf_score(capability, clean_df):
    """
    Compute the capability scores using TF-IDF.
    :param capability: capability to be searched
    :param clean_df: pandas dataframe containing cleaned text
    :return: numpy array of capability similarity scores
    """
    cleaned_capability = get_cleaned_search_phrase(capability)
    capability_sim_scores = run_tfidf_col('Capabilities', cleaned_capability, clean_df)
    capability_sim_scores_np = capability_sim_scores.flatten()
    return capability_sim_scores_np


def compute_location_score(locatn, clean_df):
    """
    Compute the distance scores.
    :param locatn: location to be searched
    :param clean_df: pandas dataframe containing cleaned text
    :return: numpy array of normalized distances
    """
    cleaned_location = get_cleaned_search_phrase(locatn)
    distances = calculate_distance(cleaned_location, clean_df)
    np_distances = np.array(distances)
    normalized_distances = normalize_values(np_distances)
    return normalized_distances


def compute_scores_single_param(k, search_queries: dict, clean_df):
    """
    Compute the scores when only one search parameter is available.
    :param k: number of top results required
    :param search_queries: dictionary, with search param as key and data to look for as value
    :param clean_df: pandas dataframe containing cleaned text
    :return: the top k ranking scores
    """
    if 'Company Name' in search_queries:
        # search for the company alone
        # do not take any other parameter into consideration
        companies = search_queries['Company Name']
        final_scores = compute_company_score(companies, clean_df)
        top_k = get_top_k(k, final_scores, clean_df)
        final_top_k = []

        for company, score, index in top_k:
            if score >= 0.6:
                final_top_k.append((company, score, index))

        return final_top_k

    if 'Capabilities' in search_queries:
        capability = search_queries['Capabilities']
        capability_scores_np = compute_capability_tfidf_score(capability, clean_df)
        final_scores = compute_ranking_fin_accr(capability_scores_np, clean_df.Accreditation, clean_df.Revenue)
        return get_top_k(k, final_scores, clean_df)

    if 'Location' in search_queries:
        locatn = search_queries['Location']
        normalized_distances = compute_location_score(locatn, clean_df)
        final_scores = compute_ranking_fin_accr(normalized_distances, clean_df.Accreditation, clean_df.Revenue)
        return get_top_k(k, final_scores, clean_df)


def compute_scores_2_param(k, search_queries: dict, clean_df):
    """
    Compute the scores when two search parameters are available.
    :param k: number of top results required
    :param search_queries: dictionary, with search param as key and data to look for as value
    :param clean_df: pandas dataframe containing cleaned text
    :return: the top k ranking scores
    """
    if 'Company Name' in search_queries and 'Capabilities' in search_queries:
        scores_dict = {}
        weights = []

        companies = search_queries['Company Name']
        company_scores = compute_company_score(companies, clean_df)
        scores_dict['Company_Score'] = company_scores
        weights.append(0.8)

        capability = search_queries['Capabilities']
        capability_scores_np = compute_capability_tfidf_score(capability, clean_df)
        scores_dict['Capability_Score'] = capability_scores_np.tolist()
        weights.append(0.2)

        weights_np = np.asarray(weights)
        scores_df = pd.DataFrame(scores_dict)
        final_scores = get_weighted_average(scores_df, weights_np, clean_df)
        return get_top_k(k, final_scores, clean_df)

    if 'Company Name' in search_queries and 'Location' in search_queries:
        scores_dict = {}
        weights = []

        companies = search_queries['Company Name']
        company_scores = compute_company_score(companies, clean_df)
        scores_dict['Company_Score'] = company_scores
        weights.append(0.7)

        locatn = search_queries['Location']
        normalized_distances = compute_location_score(locatn, clean_df)
        scores_dict['Location_Score'] = normalized_distances.tolist()
        weights.append(0.3)

        weights_np = np.asarray(weights)
        scores_df = pd.DataFrame(scores_dict)
        final_scores = get_weighted_average(scores_df, weights_np, clean_df)
        return get_top_k(k, final_scores, clean_df)

    if 'Capabilities' in search_queries and 'Location' in search_queries:
        scores_dict = {}
        weights = []

        capability = search_queries['Capabilities']
        capability_scores_np = compute_capability_tfidf_score(capability, clean_df)
        scores_dict['Capability_Score'] = capability_scores_np.tolist()
        weights.append(0.4)

        locatn = search_queries['Location']
        normalized_distances = compute_location_score(locatn, clean_df)
        scores_dict['Location_Score'] = normalized_distances.tolist()
        weights.append(0.6)

        weights_np = np.asarray(weights)
        scores_df = pd.DataFrame(scores_dict)
        final_scores = get_weighted_average(scores_df, weights_np, clean_df)
        return get_top_k(k, final_scores, clean_df)
