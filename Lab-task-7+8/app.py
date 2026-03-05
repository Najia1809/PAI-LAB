import requests
from flask import Flask,render_template,request
app = Flask(__name__)

API_KEY = "5d4b940a316044ff8744f29e0f2b75fa"

@app.route("/", methods=["GET","POST"])
def main():
    recipes = None
    if request.method =="POST":
        dish = request.form.get("dish")
        url = f"https://api.spoonacular.com/recipes/complexSearch?apiKey={API_KEY}&query={dish}&addRecipeInformation=true"
        response = requests.get(url)
        if response.status_code==200:
            data = response.json()
            recipes = data.get("results")
    return render_template("index.html",recipes=recipes)
if __name__ == "__main__":
    app.run(debug=True)


