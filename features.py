"""Dataset cleaning and feature engineering for return prediction."""

from __future__ import annotations

import pandas as pd

NUMERIC_COLUMNS = [
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
]
CATEGORICAL_COLUMNS = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [
    "Delivery_Delay_Days",
    "Customer_Return_Frequency",
]
TARGET_COLUMN = "Return_Status"
TARGET_LABELS = ("Not Returned", "Returned")


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw input columns and derive prediction features."""
    missing_columns = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required input columns: {sorted(missing_columns)}")

    frame = data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.loc[frame["Product_Price_INR"] < 0, "Product_Price_INR"] = float("nan")
    frame.loc[
        ~frame["Discount_Percent"].between(0, 100), "Discount_Percent"
    ] = float("nan")
    frame.loc[
        frame["Customer_Rating_PreOrder"].notna()
        & ~frame["Customer_Rating_PreOrder"].between(1, 5),
        "Customer_Rating_PreOrder",
    ] = float("nan")
    for column in ("Expected_Delivery_Days", "Actual_Delivery_Days"):
        frame.loc[frame[column] <= 0, column] = float("nan")
    frame.loc[frame["Previous_Returns"] < 0, "Previous_Returns"] = float("nan")
    frame.loc[frame["Previous_Orders"] <= 0, "Previous_Orders"] = float("nan")
    frame.loc[
        frame["Previous_Returns"] > frame["Previous_Orders"],
        ["Previous_Returns", "Previous_Orders"],
    ] = float("nan")

    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype("string").str.strip().replace("", pd.NA)
        frame[column] = values.astype(object).where(values.notna(), float("nan"))

    frame["Delivery_Delay_Days"] = (
        frame["Actual_Delivery_Days"] - frame["Expected_Delivery_Days"]
    )
    frame["Customer_Return_Frequency"] = (
        frame["Previous_Returns"] / frame["Previous_Orders"]
    )
    return frame[FEATURE_COLUMNS]
