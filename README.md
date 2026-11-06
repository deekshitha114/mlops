# E-Commerce Product Return Prediction

An end-to-end binary classification example using the examiner-provided CSV,
scikit-learn, FastAPI, Docker, and GitHub Actions.

## Dataset and preparation

`Section7_Q4_Ecommerce_Product_Return_CICD.csv` contains order, product,
delivery, customer-history, and `Return_Status` fields. `Order_ID` is excluded
from training because it is an identifier. Training discards rows with an
unknown target, coerces invalid numeric inputs to missing values, treats
negative prices, discounts outside 0-100%, ratings outside 1-5, nonpositive
delivery/order counts, and returns exceeding previous orders as invalid, then
imputes numeric medians and categorical modes within the training pipeline.
Categorical fields are one-hot encoded with unseen categories ignored.

Feature engineering retains the source discount percentage and adds delivery
delay (`actual - expected` days) and customer return frequency
(`previous returns / previous orders`). The held-out 20% stratified split uses
random seed 42. Logistic regression and random forest are evaluated by
accuracy, returned-class precision/recall/F1, and ROC AUC. The selected model
is the one with the highest returned-class F1 (ROC AUC breaks a tie), and is
refit on all rows before saving.

## Train and evaluate

```powershell
python -m pip install -r requirements-dev.txt
python train.py
```

The command prints the metrics and writes `artifacts/return_model.joblib` and
`artifacts/metrics.json`.

### Results on the provided dataset

The fixed-seed, stratified 20% holdout contained 240 orders. The test-set
metrics from the provided 1,200-row CSV were:

| Model | Accuracy | Returned precision | Returned recall | Returned F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.6083 | 0.2727 | 0.4444 | **0.3380** | **0.5748** |
| Random forest | **0.7125** | 0.2857 | 0.1852 | 0.2247 | 0.5318 |

Logistic regression is selected because it has the higher returned-class F1.
The low F1 and ROC AUC indicate weak separation on this dataset; this is an
educational baseline, not evidence that the model is ready for production.

## Run the API

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Check `GET /health`. Submit a `POST /predict` JSON body such as:

```json
{
  "product_category": "Electronics",
  "product_price_inr": 22055.97,
  "discount_percent": 21.6,
  "expected_delivery_days": 9,
  "actual_delivery_days": 12,
  "customer_rating_preorder": 4.4,
  "previous_returns": 0,
  "previous_orders": 33,
  "payment_method": "Card",
  "damaged_package_flag": "Yes",
  "package_size": "Large",
  "promotion_active": "Yes"
}
```

The response includes the class and estimated return probability. Invalid
request values are rejected with HTTP 422.

## Test and Docker

```powershell
ruff check .
pytest -q
docker build -t product-return-api .
docker run --rm -p 8000:8000 product-return-api
```

## GitHub Actions and GHCR

The workflow in `.github/workflows/ci-cd.yml` installs dependencies, lints,
trains/evaluates the model, and runs tests on pull requests and pushes to
`main`. A successful push to `main` builds and publishes the image as
`ghcr.io/<owner>/<repository>:latest` and tags it with the commit SHA. The
workflow uses the built-in `GITHUB_TOKEN` with package write permission; no
long-lived registry secret is needed.
