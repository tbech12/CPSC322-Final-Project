# we are going to use the flask micro web framework
# for our web app (running our API service)
import pickle
import os
from flask import Flask, jsonify, request, render_template, redirect, url_for
import mysklearn.myutils as myutils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier


# create our web app
# flask runs by default on port 5000
app = Flask(__name__,template_folder='template')

# we are now ready to setup our first "route"
# route is a function that handles a request
@app.route("/", methods =["GET", "POST"])
def index():
    # we can return content and status code
    if request.method == "POST":
        metascore = request.form.get("metascore", "") # "" is default value
        imdbrating = request.form.get("imdbrating", "")
        boxoffices_high_low = request.form.get("boxoffices", "")
        runtimes = request.form.get("runtimes", "")
        years = request.form.get("years", "")
        release_dates = request.form.get("release_dates", "")
        rated = request.form.get("rated", "")
        print("FORM:", metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated)
        return redirect(url_for(".predict", metascore=metascore, imdbrating=imdbrating, boxoffices_high_low=boxoffices_high_low,
                        runtimes=runtimes, years=years, release_dates=release_dates, rated=rated))
        #redirect("/predict?metascore={metascore}&imdbrating={imdbrating}&boxoffices={boxoffices_high_low}&runtimes={runtimes}&years={years}&release_dates={release_dates}&rated={rated}")
    return render_template('index.html'), 200

# now for the /predict endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    print(request.args.get("metascore"))
    if request.args.get("metascore") == None:
        return "Error making prediction", 400
    # parse the query string to get our
    # instance attribute values from the client
    #83.0 7.9 High 162.0 2009.0 20091218 PG-13
    metascore = request.args.get("metascore", "") # "" is default value
    imdbrating = request.args.get("imdbrating", "")
    boxoffices_high_low = request.args.get("boxoffices", "")
    runtimes = request.args.get("runtimes", "")
    years = request.args.get("years", "")
    release_dates = request.args.get("release_dates", "")
    rated = request.args.get("rated", "")

    print("args:", metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated)
    x_test = [[metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated]]
    output = prediction_classes(x_test)
    if output is not None:
        result = {"prediction": output}
        return render_template('predict.html', output=output), 200
    else:
        return "Error making prediction", 400

def prediction_classes(instance):
    file = ["dummy.pkl", "naive.pkl", "tree.pkl", "forest.pkl"]
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    forest = MyRandomForestClassifier(200, 2, 2)
    dummy = load_model(file[0])
    naive = load_model(file[1])
    tree = load_model(file[2])
    forest = load_model(file[3])

    return myutils.ensemble_helper(dummy, naive, tree, forest, instance)[0]

def check_if_all_none(list_of_elem):
    """ Check if all elements in list are None """
    result = False
    for elem in list_of_elem:
        if elem is not None:
            return True
    return result

def load_model(filename):
    infile = open(filename, "rb")
    model = pickle.load(infile)
    infile.close()
    return model

if __name__ == "__main__":
    # Deployment
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port, host="0.0.0.0") # TODO: turn debug off
    #app.run(debug=True, port=5000)
    # when deploy to "production"