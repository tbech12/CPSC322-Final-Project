# we are going to use the flask micro web framework
# for our web app (running our API service)
import pickle
import os
from flask import Flask, jsonify, request

# create our web app
# flask runs by default on port 5000
app = Flask(__name__)

# we are now ready to setup our first "route"
# route is a function that handles a request
@app.route("/", methods=["GET"])
def index():
    # we can return content and status code
    return "<h1>Welcome to my app!!</h1>", 200

# now for the /predict endpoint
@app.route("/predict", methods=["GET"])
def predict():
    # parse the query string to get our
    # instance attribute values from the client
    #83.0 7.9 High 162.0 2009.0 20091218 PG-13
    print("TRYING TO GET ARGS")
    metascore = request.args.get("metascore", "") # "" is default value
    imdbrating = request.args.get("imdbrating", "")
    boxoffices_high_low = request.args.get("boxoffices", "")
    runtimes = request.args.get("runtimes", "")
    years = request.args.get("years", "")
    release_dates = request.args.get("release_dates", "")
    rated = request.args.get("rated", "")

    print("args:", metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated)
    x_test = ['83.0', '7.9', 'High', '162.0', '2009.0', 'PG-13']
    #x_test = [[metascore, imdbrating, boxoffices_high_low, runtimes, years, release_dates, rated]]
    prediction = prediction_classes(x_test)
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

def prediction_classes(instance):
    print("GOT", instance)
    infile = open("classifiers.p", "rb")
    dummy, naive, tree, forest = pickle.load(infile)
    infile.close()
    predictions = []
    instance = [['83.0', '7.9', 'High', '162.0', '2009.0', 'PG-13'], ['50.0', '7.1', 'Mid', '169.0', '2007.0', '20070525', 'PG-13']]
    dummy_prediction = (dummy.predict(instance))
    print(dummy_prediction)
    predictions.append(dummy_prediction)
    try:
        naive_prediction = (naive.predict(instance))
        print(naive_prediction)
        predictions.append(naive_prediction)
    except:
        print("NB FAILED")
        predictions.append(None)
    try:
        tree_prediction = (tree.predict(instance))
        print(tree_prediction)
        predictions.append(tree_prediction)
    except:
        print("TREE FAILED")
        predictions.append(None)
    try:
        forest_prediction = (forest.predict(instance))
        print(forest_prediction)
        predictions.append(forest_prediction)
    except:
        print("Forest FAILED")
        predictions.append(None)


    try:
        if not all(v is None for v in predictions):
            return predictions
        else:
            print("error")
            return None
    except:
        print("error")
        return None




if __name__ == "__main__":
    # Deployment
    #port = os.environ.get("PORT", 5001)
    #app.run(debug=True, port=port, host="0.0.0.0") # TODO: turn debug off
    app.run(debug=True, port=5000)
    # when deploy to "production"