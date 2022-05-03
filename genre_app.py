"""
Programmer: Troy and James
Class: CptS 322-01, Spring 2022
Programming Final Project
"""

# we are going to use the flask micro web framework
# for our web app (running our API service)
import pickle
import os
import collections
from flask import Flask, jsonify, request, render_template, redirect, url_for
#from mysklearn.myclassifiers import MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

# create our web app
# flask runs by default on port 5000
app = Flask(__name__,template_folder='template')

# we are now ready to setup our first "route"
# route is a function that handles a request
app.route("/favicon.ico", methods=["GET"])
def error_route():
    return redirect(url_for(".index"))

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
        #print("FORM:", metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated)
        return redirect(url_for("predict", metascore=metascore, imdbrating=imdbrating, boxoffices_high_low=boxoffices_high_low,
                        runtimes=runtimes, years=years, release_dates=release_dates, rated=rated))
        #redirect("/predict?metascore={metascore}&imdbrating={imdbrating}&boxoffices={boxoffices_high_low}&runtimes={runtimes}&years={years}&release_dates={release_dates}&rated={rated}")
    elif request.method == "GET":
        return render_template('index.html'), 200
    return "ERROR", 404

# now for the /predict endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
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

    #print("args:", metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated)
    x_test = [[metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated]]
    output = prediction_classes(x_test)
    if output is not None:
        result = {"prediction": output}
        return render_template('predict.html', output=output), 200
    else:
        return "Error making prediction", 400

def prediction_classes(instance):
    """loads all prediction classes then return the predicted

    Args:
        instance (list): list of arguements

    Returns:
        string: returns the predicted value
    """
    file = ["dummy.pkl", "naive.pkl", "tree.pkl", "forest.pkl"]
    dummy = load_model(file[0])
    naive = load_model(file[1])
    tree = load_model(file[2])
    forest = load_model(file[3])

    return ensemble_helper(dummy, naive, tree, forest, instance)[0]

def ensemble_helper(dummy, nb, tree, forest, X_test):
    """_summary_

    Args:
        dummy (class): dummy classifier
        nb (class): nb classifier
        tree (class): tree classifier
        forest (class): radnom forest classifier
        X_test (list): items to predict

    Returns:
        list: list of 1 object
    """
    strat_ensemble = []
    for test in X_test:
        prediction = []
        try:
            prediction.extend(nb.predict([test])) #predicts
        except:
            prediction.append(None)

        try:
            prediction.extend(dummy.predict([test]))#predicts
        except:
            prediction.append(None)

        try:
            prediction.extend(tree.predict([test]))#predicts
        except:
            prediction.append(None)

        try:
            prediction.extend(forest.predict([test]))#predicts
        except:
            prediction.append(None)

        #gets most frquest
        most_freq = [item for item, count in collections.Counter(prediction).items() if count > 1]
        if most_freq == []:
            strat_ensemble.append(prediction[0]) #if empty get prediction first
        elif len(most_freq) > 1:
            strat_ensemble.append(most_freq[0]) #if getreater than 1 gets first
        else:
            strat_ensemble.append(most_freq[0]) #else just gets first
    return strat_ensemble

def check_if_all_none(list_of_elem):
    """ Check if all elements in list are None """
    result = False
    for elem in list_of_elem:
        if elem is not None:
            return True
    return result

def load_model(filename, file_prefix="pickled_objects/"):
    """loads the classifiers

    Args:
        filename (str): file name
        file_prefix (str, optional): file prefix. Defaults to "pickled_objects/".

    Returns:
        pickled object: classifiers
    """
    infile = open(file_prefix + filename, "rb")
    model = pickle.load(infile)
    infile.close()
    return model

if __name__ == "__main__":
    # Deployment
    #port = os.environ.get("PORT", 5000)
    #app.run(debug=True, port=port, host="0.0.0.0") # TODO: turn debug off
    app.run(debug=True, port=5000)
    # when deploy to "production"