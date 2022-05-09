from flask import Flask
from flask import request
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import url_for
from Server import *
app=Flask(__name__,static_folder="./templates")

filePath = r"./Data_AM_Finance_Accreditation.xlsx"
cleanDF = clean_dataframe(filePath)

@app.route("/")
def first_page():
    return 'this is the first page'

@app.route('/TF_IDF_model/<searchPhrase>')
def use_TF_IDF_model(searchPhrase):
    """
    Args:
        searchPhrase: Text format, user-entered keywords
    Returns:
        Json format,that output the top five recommend company by TF_IDF algorithm
    """
    print('use tf_idf')
    if searchPhrase == False and searchPhrase.isspace() == False:
        return 'error'
    info=run_tfidf_vectorizer_model(5,searchPhrase,cleanDF)
    # get top 5 scores index in ascending order
    final={}
    count=1
    # print(info)
    for name,score,index in info:
        temp=cleanDF.loc[index,:].to_dict()
        temp['score'] = score
        data='company'+str(count)
        final[data]=temp
        count+=1
    return jsonify(final)

@app.route('/SBERT_model/<searchPhrase>')
def use_SBERT_model(searchPhrase):
    """
    Args:
        searchPhrase: Text format, user-entered keywords
    Returns:
        Json format,that output the top five recommend company by SBERT_model algorithm
    """
    print("use sbert")
    if searchPhrase == False and searchPhrase.isspace() == False:
        return 'error'
    top_results=run_sbert_model(5,searchPhrase,cleanDF)
    # get top 5 scores index in ascending order
    final={}
    count=1
    for score, index in zip(top_results[0], top_results[1]):
        sco=float(score)
        idx = int(index)
        temp=cleanDF.loc[idx,:].to_dict()
        temp['score'] = sco
        data='company'+str(count)
        final[data]=temp
        count+=1
    return jsonify(final)

@app.route('/filters_model/<search_queries>')
def use_filters_model(search_queries:dict):
    """
    Args:
        searchPhrase: dictionary format, user-entered keywords
    Returns:
        Json format,that output the top five recommend company
    """

    results = run_filters_model(5, search_queries, cleanDF)

    # get top 5 scores index in ascending order
    final={}
    count=1
    # print(info)
    for name,score,index in results:
        temp=cleanDF.loc[index,:].to_dict()
        temp['score'] = score
        data='company'+str(count)
        final[data]=temp
        count+=1
    return jsonify(final)


@app.route('/index',methods=['GET','POST'])
def index():
    """
        The user interface page, user could use this page for inputting key worlds and choice algorithm
    Returns:
        use function of use_SBERT_model or use_SBERT_model
    """
    if request.method=="GET":
        return render_template("index.html")
    else:
        searchPhrase=request.values.get("keywords")
        method=request.values.get("method")
        if(method=='SBERT'):
            return redirect(url_for('use_SBERT_model',searchPhrase=searchPhrase))
        else:
            return redirect(url_for('use_TF_IDF_model',searchPhrase=searchPhrase))

@app.route('/index_filter',methods=['GET','POST'])
def index2():
    """
        The user interface page
    """

    if (request.method=="GET"):
        return render_template("index2.html")
    else:

        company_name=request.values.get("company_name")
        capability=request.values.get("capability")
        location=request.values.get("location")
        search_phrases: dict[str, str] = {'Location': location, 'Capabilities':capability ,
                                          'Company Name': company_name}
        cleaned_phrases =search_phrases.copy()
        for k,v in search_phrases.items():
            if not v:
                del cleaned_phrases[k]

        if(cleaned_phrases):
            filters_result=run_filters_model(5, search_phrases, cleanDF)

            final = {}
            count = 1
            # print(info)
            for name, score, index in filters_result:
                temp = cleanDF.loc[index, :].to_dict()
                temp['score'] = score
                data = 'company' + str(count)
                final[data] = temp
                count += 1
            return jsonify(final)
        else:
            return "please input the params"

if __name__=="__main__":
    app.run(host="0.0.0.0")
    # print(use_TF_IDF_model("laser"))

