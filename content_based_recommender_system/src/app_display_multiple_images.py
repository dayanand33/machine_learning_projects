import os

from amazon_recommender_models import AllModelsInitializer
from flask import Flask, request, render_template, send_from_directory

__author__ = 'ibininja'

app = Flask(__name__)



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

all_models = AllModelsInitializer()
all_models.initialize()

@app.route("/")
@app.route('/home', methods = ['GET'])
def home():
    return render_template('form2.html')

@app.route('/submit-form', methods = ['POST'])
def submitForm():
    select_value = request.form.get('select1')
    print('form submitted with value {select_value}')
    return(str(select_value))

@app.route('/recommender' , methods=["POST"])
def generate_recommendations():
    doc_id = int(request.form.get('doc_id'))
    model_name = request.form.get('model_name')
    #doc_id=0
    #model_name = 'weighted_w2v_model'
    print(f'doc id is {doc_id}')
    print(f'model_name is {model_name}')


    results = None
    if model_name == 'bag_of_words':
        results = all_models.bag_of_words.get_similar_products(doc_id)
    elif model_name == 'tf_idf':
        results = all_models.tf_idf.get_similar_products(doc_id)
    elif model_name == 'idf':
        results = all_models.idf_model.get_similar_products(doc_id)
    elif model_name == 'idf_w2v_brand':
        results = all_models.word_vec_model.idf_w2v_brand(doc_id, 5, 5)
    elif model_name == 'avg_w2v_model':
        results = all_models.word_vec_model.avg_w2v_model(doc_id)
    elif model_name == 'weighted_w2v_model':
        results = all_models.word_vec_model.weighted_w2v_model(doc_id)


    image_urls = []
    titles = []
    for result in results:
        image_urls.append(result[1])
        titles.append(result[0])
    return render_template("gallery2.html", image_urls=image_urls, titles= titles)

#generate_recommendations()
if __name__ == "__main__":
    app.run(port=4555, debug=True)