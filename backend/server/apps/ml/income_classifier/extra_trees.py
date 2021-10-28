import joblib
import pandas as pd
from pandas._libs import missing

class ExtraTreesClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.values_fill_missing = joblib.load(path_to_artifacts + "train_mode.joblib")
        self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "extra_trees.joblib")

    def preprocessing(self, input_data):
        # convert json to pandas df
        input_data = pd.DataFrame(input_data, index=[0])
        # filling missing values
        input_data.fillna(self.values_fill_missing)
        # convert categorical values

        for column in  [
            "workplace",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]:
            categorical_convert = self.encoders[column]
            input_data[column] = categorical_convert.fit_transform(input_data[column])
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = "<=50K"
        if input_data[1] > 0.5:
            label = ">50K"
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": repr(e)}

        return prediction
