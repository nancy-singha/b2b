from dao import *
import os


def clean_dataframe(file_path):
    """
    Cleans the dataset.
    :param file_path: path of the file containing input data
    :return: pandas DataFrame containing cleaned data
    """
    random.seed(42)
    df = pd.read_excel(file_path, engine='openpyxl')

    company = df['CompanyName']
    capabilities = df['Capabilities']
    location = df['DefaultAddress']
    accr = df['Accreditation']
    website = df['WebSite']
    description = df['CompanyDescription']
    interests = df['Interests']
    products = df['Products']
    projects = df['Projects']
    finance_info = df['NetRevenue']
    latitudes = df['lat'].astype(str)
    longitudes = df['lng'].astype(str)

    websiteVocab = url_cleaning(website)
    descriptionVocab, descr = perform_texts_clean(description, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False)
    # Makes sense to have stemming for capabilities, as it contains verbs
    capabilitiesVocab, capab = perform_texts_clean(capabilities, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False)
    # DefaultAddress should not have stemming but lemmatising
    locationVocab, loca = perform_texts_clean(location, "[^-0-9A-Za-z ]", do_lemmatize=False, do_stemming=False)
    accrVocab, accre = perform_texts_clean(accr, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False)
    interestsVocab, interes = perform_texts_clean(interests, REGEX_BASIC_CONST, do_lemmatize=True, do_stemming=False)
    # Products contain verbs, hence perform stemming
    prodVocab, prod = perform_texts_clean(products, REGEX_ADV_CONST, do_lemmatize=True, do_stemming=False)
    projVocab, proj = perform_texts_clean(projects, REGEX_BASIC_CONST, do_lemmatize=True, do_stemming=False)
    finance_bins = convert_to_bins(finance_info)
    comp = basic_text_cleaning(company)

    cleaned_columns = {'Comp': comp, 'description': descr, 'Capabilities': capab,
                       'Location': loca, 'Accreditation': accre, 'Revenue': finance_bins, 'Products': prod,
                       'Projects': proj, 'latitudes': latitudes, 'longitudes': longitudes}
    clean_df = pd.DataFrame(cleaned_columns)
    return clean_df


def run_tfidf_vectorizer_model(k, query, clean_df):
    """
    Run TF-IDF model.
    :param k: number of top results required
    :param query: the phrase to look for, could be a sentence
    :param clean_df: pandas data frame containing cleaned text
    :return: list of tuples containing company name, score and index of supplier
    """
    fitted_data = fit_vectorizer(clean_df)
    cleaned_query = get_cleaned_search_phrase(query)
    search_data = predict_search_vector(cleaned_query)
    score_data = get_top_k_scores(fitted_data, search_data, k, True, clean_df)
    return score_data


def run_sbert_model(k, query, clean_df):
    """
    Run SBERT model.
    :param k: number of top results required
    :param query: search query as a sentence
    :param clean_df: pandas data frame containing cleaned text
    :return: the top k ranking scores
    """
    sentence = clean_df.apply(' '.join, axis=1)
    sentences = sentence.values.tolist()
    search_sentence = get_cleaned_search_phrase(query)
    return use_sentence_embedder(sentences, search_sentence, clean_df, k)


def run_filters_model(k, search_queries: dict, clean_df):
    """
    Model to get ranking for multiple queries.
    :param k: number of top results required
    :param search_queries: dictionary, with search param as key and data to look for as value
    :param clean_df : that included the data
    :return: list of tuples containing company name, score and index of supplier
    """
    print("Use Filters")
    file_path = os.path.join(".", "..", "..", "data", "company_address_finance.xlsx")
    clean_df = clean_dataframe(file_path)

    num_params = len(search_queries)

    if num_params == 1:
        return compute_scores_single_param(k, search_queries, clean_df)

    if num_params == 2:
        return compute_scores_2_param(k, search_queries, clean_df)

    if num_params == 3:
        scores_dict = {}
        weights = []

        companies = search_queries['Company Name']
        company_scores = compute_company_score(companies, clean_df)
        scores_dict['Company_Score'] = company_scores
        weights.append(0.5)

        capability = search_queries['Capabilities']
        capability_scores_np = compute_capability_tfidf_score(capability, clean_df)
        scores_dict['Capability_Score'] = capability_scores_np.tolist()
        weights.append(0.3)

        locatn = search_queries['Location']
        normalized_distances = compute_location_score(locatn, clean_df)
        scores_dict['Location_Score'] = normalized_distances.tolist()
        weights.append(0.2)

        weights_np = np.asarray(weights)
        scores_df = pd.DataFrame(scores_dict)
        final_scores = get_weighted_average(scores_df, weights_np, clean_df)
        return get_top_k(k, final_scores, clean_df)
