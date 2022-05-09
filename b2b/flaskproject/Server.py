from Dao import *


def clean_dataframe(filePath):
    """
    Args:
        filePath: it is the file path
    Returns:
        The cleaned DataFrame
    """
    random.seed(42)

    df = pd.read_excel(filePath, engine='openpyxl')
    company = df['Company Name']
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

    REGEX_ADV_CONST = "[^-:,0-9A-Za-z ]"
    REGEX_BASIC_CONST = "[^-,0-9A-Za-z ]"

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



    cleanedColumns = {'Comp': comp, 'description': descr, 'Capabilities': capab,
                      'Location': loca,'Accreditation': accre, 'Revenue': finance_bins, 'Products': prod,
                      'Projects': proj,'latitudes':latitudes,'longitudes':longitudes}
    cleanDF = pd.DataFrame(cleanedColumns)
    return cleanDF
def run_tfidf_vectorizer_model(k, query,cleanDF):
    """
    Run TF-IDF model.
    :param k: number of top results required
    :param query: the phrase to look for, could be a sentence
    :return: list of tuples containing company name, score and index of supplier
    """
    fitted_data = fit_vectorizer(cleanDF)
    cleaned_query = get_cleaned_search_phrase(query)
    search_data = predict_search_vector(cleaned_query)
    score_data = get_top_k_scores(fitted_data, search_data, k, True,cleanDF)

    # for companyName, scores, index in score_data:
    #     print('Company {0} with score {1}, and index is {2}'.format(companyName, scores, index))
    return score_data
def run_sbert_model(k, query,cleanDF):
    """
    Run SBERT model.
    :param k: number of top results required
    :param query: search query as a sentence
    :return: the top k ranking scores
    """

    sentence = cleanDF.apply(' '.join, axis=1)
    sentences = sentence.values.tolist()
    search_sentence = get_cleaned_search_phrase(query)
    return use_sentence_embedder(sentences, search_sentence,cleanDF,k)
def run_filters_model(k,search_queries: dict,cleanDF):
    """
    Model to get ranking for multiple queries.
    :param k: number of top results required
    :param search_queries: dictionary, with search param as key and data to look for as value
    :param cleanDF : that included the data
    :return: list of tuples containing company name, score and index of supplier
    """
    print("Use Filters")

    filePath = r'./Data_AM_Finance_Accreditation.xlsx'
    cleanDF = clean_dataframe(filePath)

    num_params = len(search_queries)

    if num_params == 1:
        return compute_scores_single_param(k, search_queries,cleanDF)

    if num_params == 2:
        return compute_scores_2_param(k, search_queries,cleanDF)

    if num_params == 3:
        scores_dict = {}
        weights = []

        companies = search_queries['Company Name']
        company_scores = compute_company_score(companies,cleanDF)
        scores_dict['Company_Score'] = company_scores
        weights.append(0.5)

        capability = search_queries['Capabilities']
        capability_scores_np = compute_capability_tfidf_score(capability,cleanDF)
        scores_dict['Capability_Score'] = capability_scores_np.tolist()
        weights.append(0.3)

        locatn = search_queries['Location']
        normalized_distances = compute_location_score(locatn,cleanDF)
        scores_dict['Location_Score'] = normalized_distances.tolist()
        weights.append(0.2)

        weights_np = np.asarray(weights)
        scores_df = pd.DataFrame(scores_dict)
        final_scores = get_weighted_average(scores_df, weights_np,cleanDF)
        return get_top_k(k, final_scores,cleanDF)
