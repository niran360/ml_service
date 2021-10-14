import joblib
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts = "../../research"
        self.values_fill_missing = joblib.load(path_to_artifacts + "train_mode.joblib")
        self.encoders = joblib.load(path_to_artifacts +"encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest.joblib")

    def preprocessing(self, input_data):
        #convert JSON to pandas Dataframe
        input_data = pd.DataFrame(input_data, index=[0])
        #fill missing values
        input_data.fillna(self.values_fill_missing)
        # convert categorical values
        for columns in [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]:
            categorical_convert = self.encoders[column]
            input_data[column] =categorical_convert,transform(input_data[column])
        
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def preprocessing(self, input_data):
        label = "<=50"
        if input_data[1] > 0.5:
            label = ">50k"
        return {"probability": input_data[1], "label" :label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]
            prediction = self.preprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction