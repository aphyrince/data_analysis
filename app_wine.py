from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd

app = Flask(__name__)
app.config["RESTX_JSON"] = {"ensure_ascii": False}
api = Api(
    app,
    version="1.0",
    title="Wine API",
    description="성분 13 가지를 받아 품종을 돌려준다",
    doc="/api/docs",
    prefix="/api",
)
ns = api.namespace("wine", description="wine 품종 예측")
# 서버를 켤 때 한 번만 읽는다
model = joblib.load("wine_model.pkl")
COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]
EXAMPLE = [
    13.74,
    1.67,
    2.25,
    16.4,
    118.0,
    2.6,
    2.9,
    0.21,
    1.62,
    5.85,
    0.92,
    3.2,
    1060.0,
]
NAMES = ["class_0", "class_1", "class_2"]

flower = api.model(
    "Flower",
    {
        "sepal_length": fields.Float(
            required=True, example=6.0, description="꽃받침 길이 (cm)"
        ),
        "sepal_width": fields.Float(
            required=True, example=2.9, description="꽃받침 너비 (cm)"
        ),
        "petal_length": fields.Float(
            required=True, example=4.5, description="꽃잎 길이 (cm)"
        ),
        "petal_width": fields.Float(
            required=True, example=1.5, description="꽃잎 너비 (cm)"
        ),
    },
)

# 열 13 개를 하나씩 적지 않고 COLUMNS 와 EXAMPLE 에서 만든다
wine_in = api.model(
    "Wine",
    {c: fields.Float(required=True, example=e) for c, e in zip(COLUMNS, EXAMPLE)},
)


@ns.route("/predict")
class Predict(Resource):
    @ns.doc("품종 예측")
    @ns.expect(wine_in, validate=True)
    def post(self):
        """성분 13 가지를 받아 품종과 확률을 돌려준다"""
        d = api.payload
        x = pd.DataFrame([[d[c] for c in COLUMNS]], columns=COLUMNS)
        proba = model.predict_proba(x)[0]
        i = int(proba.argmax())
        return {"kind": NAMES[i], "probability": round(float(proba[i]), 2)}
