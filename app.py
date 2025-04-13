from flask import Flask, render_template, request
from recommender import get_recommendations

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        preference = request.form.get('preference')
        recommendations = get_recommendations(preference)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)