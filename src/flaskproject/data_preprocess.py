import pandas as pd
import os
import requests
import json
import fuzzymatcher
import finance_data_collector as finB4S


def address_update(company_data):
    """
    Update geographical co-ordinates for all addresses.
    :param company_data: company details
    :return: updated company_data
    """
    for i, row in company_data.iterrows():
        api_address = str(company_data.at[i, 'DefaultAddress']) + ',' + str(company_data.at[i, 'cityState']) + ',' + str(
            company_data.at[i, 'PinCode']) + ',' + str(company_data.at[i, 'Country'])

        parameters = {
            "key": "Bsr2Opqfz9wS6239vUEdXIPlbCf1lAte",
            "location": api_address
        }

        response = requests.get("http://www.mapquestapi.com/geocoding/v1/address", params=parameters)
        data = response.text
        data_json = json.loads(data)['results']
        lat = (data_json[0]['locations'][0]['latLng']['lat'])
        lng = (data_json[0]['locations'][0]['latLng']['lng'])

        company_data.at[i, 'lat'] = lat
        company_data.at[i, 'lng'] = lng

    return company_data


def company_finance(input_dataset):
    """
    Update financial data in the original dataset
    :param input_dataset: input data
    :return: pandas DataFrame containing financial details
    """
    print("Invoked financial data collection")
    # Extract Financial data using BeautifulSoup
    finB4S.extract_finance_data()

    # read the Json file with financial results
    path = os.path.join(".", "..", "..")
    result_folder = os.path.join(path, "monthly_results")
    bS4_result = os.path.join(result_folder, "result.json")

    text = open(bS4_result).read()
    text = text.replace(', null', '')
    text = text.replace('["', '')
    text = text.replace('{}', '"0"')
    text = text.replace('[', '')
    lis = text.split('],')
    dct = {i: j.split('", "') for i, j in enumerate(lis)}
    df = pd.DataFrame.from_dict(dct, orient='index')
    df = df.rename(columns={0: 'CompanyName1', 1: 'CompanyRegistration', 2: 'NetRevenue'})
    df = df.fillna(value="")
    df['CompanyName1'] = df['CompanyName1'].apply(lambda x: x.replace(',', '')).apply(lambda x: x.upper())
    df['CompanyRegistration'] = df['CompanyRegistration'].apply(lambda x: x.replace(',', '')).apply(lambda x: x.upper())
    df['NetRevenue'] = df['NetRevenue'].apply(lambda x: x.replace('"', '')).apply(lambda x: x.replace(']', '')).apply(
        lambda x: x.replace(',', '')).apply(lambda x: x.replace(' ', ''))

    input_dataset['CompanyName'] = input_dataset['CompanyName'].apply(lambda x: x.upper())

    # Apply fuzzy matching on Company name
    left_on = ['CompanyName']
    right_on = ['CompanyName1']
    matched_results = fuzzymatcher.fuzzy_left_join(input_dataset,
                                                   df,
                                                   left_on,
                                                   right_on,
                                                   left_id_col='CompanyID',
                                                   right_id_col='CompanyRegistration'
                                                   )

    cols = ["CompanyID", "NetRevenue", 'best_match_score']
    matched_results.to_csv('fuzzy_res.csv')
    fuzzy_result = matched_results
    fuzzy_result['rank'] = matched_results.groupby("CompanyID")["best_match_score"].rank(ascending=False)
    fuzzy_result = fuzzy_result[fuzzy_result['rank'] == 1]
    fuzzy_result = fuzzy_result[cols].query("best_match_score >= .50").sort_values(by=['best_match_score'],
                                                                                   ascending=False)

    finance = pd.merge(input_dataset, fuzzy_result, on='CompanyID', how='left')
    finance['NetRevenue'] = finance['NetRevenue'].fillna(0)
    finance['NetRevenue'] = finance['NetRevenue'].astype(float)
    return finance.iloc[:, :-1]


def data_preprocessing():
    """
    Perform input data pre processing.
    :return: None
    """
    path = os.path.join(".", "..", "..", "data")
    filepath = os.path.join(path, "Data_AM_Accreditation(Address).xlsx")
    input_dataset = pd.read_excel(filepath, index_col=None, header=0, engine='openpyxl')
    input_dataset = input_dataset.fillna('NA')

    company_data = address_update(input_dataset)
    company_data_path = os.path.join(path, "company_address.xlsx")
    company_data.to_excel(company_data_path)

    # pull Financial information for companies:
    result = company_finance(company_data)
    company_finance_path = os.path.join(path, "company_address_finance.xlsx")
    result.to_excel(company_finance_path)
