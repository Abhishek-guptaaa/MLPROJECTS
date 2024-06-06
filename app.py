from flask import Flask, render_template, request
from src.mlproject.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Extract data from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )
            # Prepare data for prediction
            pred_df = data.get_data_as_data_frame()
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=results[0])
        except Exception as e:
            # Handle errors gracefully
            error_message = f"An error occurred: {str(e)}"
            return render_template('home.html', error_message=error_message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
