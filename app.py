"""FastAPI application for product return predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from features import engineer_features

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "return_model.joblib"
MODEL = joblib.load(MODEL_PATH)

app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    version="1.0.0",
    description="Predicts whether an e-commerce order is likely to be returned.",
)


class PredictionRequest(BaseModel):
    product_category: str = Field(min_length=1)
    product_price_inr: float = Field(ge=0)
    discount_percent: float = Field(ge=0, le=100)
    expected_delivery_days: float = Field(gt=0)
    actual_delivery_days: float = Field(gt=0)
    customer_rating_preorder: float = Field(ge=1, le=5)
    previous_returns: int = Field(ge=0)
    previous_orders: int = Field(gt=0)
    payment_method: str = Field(min_length=1)
    damaged_package_flag: Literal["Yes", "No"]
    package_size: Literal["Small", "Medium", "Large"]
    promotion_active: Literal["Yes", "No"]

    @model_validator(mode="after")
    def returns_cannot_exceed_orders(self) -> PredictionRequest:
        if self.previous_returns > self.previous_orders:
            raise ValueError("previous_returns cannot exceed previous_orders")
        return self


class PredictionResponse(BaseModel):
    prediction: Literal["Returned", "Not Returned"]
    return_probability: float
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    values = request.model_dump()
    raw = pd.DataFrame(
        [
            {
                "Product_Category": values["product_category"],
                "Product_Price_INR": values["product_price_inr"],
                "Discount_Percent": values["discount_percent"],
                "Expected_Delivery_Days": values["expected_delivery_days"],
                "Actual_Delivery_Days": values["actual_delivery_days"],
                "Customer_Rating_PreOrder": values["customer_rating_preorder"],
                "Previous_Returns": values["previous_returns"],
                "Previous_Orders": values["previous_orders"],
                "Payment_Method": values["payment_method"],
                "Damaged_Package_Flag": values["damaged_package_flag"],
                "Package_Size": values["package_size"],
                "Promotion_Active": values["promotion_active"],
            }
        ]
    )
    features = engineer_features(raw)
    prediction = int(MODEL.predict(features)[0])
    probability = float(MODEL.predict_proba(features)[0, 1])
    return PredictionResponse(
        prediction="Returned" if prediction else "Not Returned",
        return_probability=probability,
        model="selected-by-held-out-f1",
    )
