from operator import concat
import re
import json
from bs4 import BeautifulSoup
import shutil
import os


def process_html(fname, result_file):
    gmp = open(fname, 'r', encoding='utf-8')
    name = {}
    number = {}
    asset = {}
    soup = BeautifulSoup(gmp, 'html.parser')
    n = r = c = 0

    for item in soup.find_all(["ix:nonfraction", "ix:nonnumeric"]):  # Match soup tag name here
        init = item['name'].split(':')[0]

        # Net Asset

        if re.match(concat(init, ":NetAssetsLiabilities"), item['name']):  # Match HTML tag name here
            n = n + 1
            if n == 1:
                asset = item.get_text()

        elif re.match(concat(init, ":NetAssets(Liabilities)"), item['name']):  # Match HTML tag name here
            n = n + 1
            if n == 1:
                asset = item.get_text()
        elif re.match(concat(init, ":TotalAssetsLessCurrentLiabilities"), item['name']):
            n = n + 1
            if n == 1:
                asset = item.get_text()
        elif re.match(concat(init, ":NetCurrentAssetsLiabilities"), item['name']):  # Match HTML tag name here
            n = n + 1
            if n == 1:
                asset = item.get_text()

        # Name
        if re.match(concat(init, ":EntityCurrentLegalOrRegisteredName"), item['name']):
            c = c + 1
            if c == 1:
                name = item.get_text()

        # Registered Number
        if re.match(concat(init, ":UKCompaniesHouseRegisteredNumber"), item['name']):
            r = r + 1
            if r == 1:
                number = item.get_text()

    result = [name, number, asset]
    gmp.close()
    if n > 0:
        return result


def extract_finance_data():
    """
    Extract financial data using beautifulSoup.
    :return: None
    """
    # Place html files folder in the same folder with python file
    filepath = os.path.join(".", "..", "..")
    source_folder = os.path.join(filepath, "accounts_monthly_data-march2022")
    archive_folder = os.path.join(filepath, "monthly_archive")
    result_folder = os.path.join(filepath, "monthly_results")

    url_directory = os.path.join(source_folder)  # keep all files in one folder separate from code file
    filename = os.listdir(url_directory)
    dataset_length = len(filename)
    file_counter = 0
    print('Files to be processed: ', dataset_length)
    result = []
    if dataset_length > 0:
        while file_counter < dataset_length:  # process files in 50K batches
            result_file = os.path.join(result_folder, "result.json")
            fname = os.path.join(source_folder, filename[file_counter])
            res = process_html(fname, result_file)
            if res != [{}, {}, {}]:
                result.append(res)
                shutil.move(fname, archive_folder)  # archive files
            file_counter = file_counter + 1

            if file_counter % 40000 == 0:
                print(file_counter, ' files processed')

        if file_counter == dataset_length:
            with open(result_file, 'w') as storage_file:
                storage_file.write(json.dumps(result))
            print('created file:', result_file)
