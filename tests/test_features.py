import math

import pandas as pd

from features import engineer_features


def test_feature_engineering_cleans_invalid_values_and_derives_features() -> None:
    raw = pd.DataFrame(
        [
            {
                "Product_Category": " Electronics ",
                "Product_Price_INR": -10,
                "Discount_Percent": 125,
                "Expected_Delivery_Days": 5,
                "Actual_Delivery_Days": 8,
                "Customer_Rating_PreOrder": 4,
                "Previous_Returns": 2,
                "Previous_Orders": 10,
                "Payment_Method": "Card",
                "Damaged_Package_Flag": "No",
                "Package_Size": "Medium",
                "Promotion_Active": "Yes",
            }
        ]
    )

    features = engineer_features(raw)

    assert math.isnan(features.loc[0, "Product_Price_INR"])
    assert math.isnan(features.loc[0, "Discount_Percent"])
    assert features.loc[0, "Delivery_Delay_Days"] == 3
    assert features.loc[0, "Customer_Return_Frequency"] == 0.2
    assert features.loc[0, "Product_Category"] == "Electronics"
