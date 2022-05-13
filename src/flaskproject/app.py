from flask import Flask
from flask import request
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import url_for
from server import *
import os

app = Flask(__name__, static_folder="./templates")

# filePath = r"./Data_AM_Finance_Accreditation.xlsx"
filePath = r"./../../data/company_address_finance.xlsx"
cleanDF = clean_dataframe(filePath)


@app.route("/")
def first_page():
    return 'this is the first page'


@app.route('/TF_IDF_model/<search_phrase>')
def use_tf_idf_model(search_phrase):
    """
    Args:
        search_phrase: Text format, user-entered keywords
    Returns:
        Json format,that output the top five recommend company by TF_IDF algorithm
    """
    print('use tf_idf')
    if search_phrase is False and search_phrase.isspace() is False:
        return 'error'
    info = run_tfidf_vectorizer_model(5, search_phrase, cleanDF)
    # get top 5 scores index in ascending order
    final = {}
    count = 1
    # print(info)
    for name, score, idx in info:
        temp = cleanDF.loc[idx, :].to_dict()
        temp['score'] = score
        data = 'company' + str(count)
        final[data] = temp
        count += 1
    return jsonify(final)


@app.route('/SBERT_model/<search_phrase>')
def use_sbert_model(search_phrase):
    """
    Args:
        search_phrase: Text format, user-entered keywords
    Returns:
        Json format,that output the top five recommend company by SBERT_model algorithm
    """
    print("use SBERT")
    if search_phrase is False and search_phrase.isspace() is False:
        return 'error'
    top_results = run_sbert_model(5, search_phrase, cleanDF)
    # get top 5 scores index in ascending order
    final = {}
    count = 1
    for score, data_index in zip(top_results[0], top_results[1]):
        sco = float(score)
        idx = int(data_index)
        temp = cleanDF.loc[idx, :].to_dict()
        temp['score'] = sco
        data = 'company' + str(count)
        final[data] = temp
        count += 1
    return jsonify(final)


@app.route('/filters_model/<search_queries>')
def use_filters_model(search_queries: dict):
    """
    Args:
        search_queries: dictionary format, user-entered keywords
    Returns:
        Json format,that output the top five recommend company
    """

    results = run_filters_model(5, search_queries, cleanDF)

    # get top 5 scores index in ascending order
    final = {}
    count = 1
    # print(info)
    for name, score, idx in results:
        temp = cleanDF.loc[idx, :].to_dict()
        temp['score'] = score
        data = 'company' + str(count)
        final[data] = temp
        count += 1
    return jsonify(final)


@app.route('/index', methods=['GET', 'POST'])
def index():
    """
        The user interface page, user could use this page for inputting key worlds and choice algorithm
    Returns:
        use function of use_SBERT_model or use_SBERT_model
    """
    if request.method == "GET":
        return render_template("index.html")
    else:
        search_phrase = request.values.get("keywords")
        method = request.values.get("method")
        if method == 'SBERT':
            return redirect(url_for('use_sbert_model', search_phrase=search_phrase))
        else:
            return redirect(url_for('use_tf_idf_model', search_phrase=search_phrase))


@app.route('/index_filter', methods=['GET', 'POST'])
def index2():
    """
        The user interface page
    """

    if request.method == "GET":
        return render_template("index2.html")
    else:

        company_name = request.values.get("company_name")
        capability = request.values.get("capability")
        location = request.values.get("location")
        search_phrases: dict[str, str] = {'Location': location, 'Capabilities': capability,
                                          'Company Name': company_name}
        cleaned_phrases = search_phrases.copy()
        for k, v in search_phrases.items():
            if not v:
                del cleaned_phrases[k]

        if cleaned_phrases:
            filters_result = run_filters_model(5, cleaned_phrases, cleanDF)

            final = {}
            count = 1
            for name, score, idx in filters_result:
                temp = cleanDF.loc[idx, :].to_dict()
                temp['score'] = score
                data = 'company' + str(count)
                final[data] = temp
                count += 1
            return jsonify(final)
        else:
            return "please input the params"


if __name__ == "__main__":
    # app.run(host="0.0.0.0")
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)))
    # print(use_TF_IDF_model("laser"))
