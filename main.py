import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
# ============================================================
# LOAD ENVIRONMENT
# ============================================================
load_dotenv()
app = FastAPI(title="Supply Chain Stock Prediction API", version="1.0.0")
# ============================================================
# S/4HANA CONFIGURATION
# ============================================================
S4_BASE_URL = os.getenv("S4_BASE_URL", "http://192.168.8.69:50000").rstrip("/")
S4_USERNAME = os.getenv("S4_USERNAME")
S4_PASSWORD = os.getenv("S4_PASSWORD")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"
# ============================================================
# SAP AI CORE / TABPFN CONFIGURATION
# ============================================================
AICORE_AUTH_URL = os.getenv("AICORE_AUTH_URL")
AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID")
AICORE_CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET")
AICORE_API_URL = os.getenv("AICORE_API_URL")
TABPFN_DEPLOYMENT_ID = os.getenv("TABPFN_DEPLOYMENT_ID")
AICORE_RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP", "default")
# ============================================================
# S/4 SERVICE PATHS
# ============================================================
STOCK_SERVICE = "/sap/opu/odata/sap/" "API_MATERIAL_STOCK_SRV"
MATERIAL_DOCUMENT_SERVICE = "/sap/opu/odata/sap/" "API_MATERIAL_DOCUMENT_SRV"
PURCHASE_ORDER_SERVICE = "/sap/opu/odata/sap/" "API_PURCHASEORDER_PROCESS_SRV"
SALES_ORDER_SERVICE = "/sap/opu/odata/sap/" "API_SALES_ORDER_SRV"
# ============================================================
# TABPFN FEATURES
# ============================================================
FEATURES = ["CurrentStock", "Consumption30D", "IncomingPO", "Demand"]
TARGET = "StockAfter7Days"
# ============================================================
# REQUEST MODEL
# ============================================================
class PredictionRequest(BaseModel):
    material: str
    plant: str
# ============================================================
# HTTP SESSION FOR S/4
# ============================================================
s4_session = requests.Session()
s4_session.auth = HTTPBasicAuth(S4_USERNAME, S4_PASSWORD)
s4_session.headers.update({"Accept": "application/json"})
# ============================================================
# HELPER
# ============================================================
def safe_float(value):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
def escape_odata(value):
    return str(value).replace("'", "''")
# ============================================================
# GENERIC ODATA CALL
# ============================================================
def odata_get(service, entity, params=None):
    url = f"{S4_BASE_URL}" f"{service}/" f"{entity}"
    response = s4_session.get(url, params=params, verify=VERIFY_SSL, timeout=120)
    response.raise_for_status()
    data = response.json()
    # OData V2
    if "d" in data:
        result = data["d"]
        if isinstance(result, dict):
            if "results" in result:
                return result["results"]
            return [result]
    # OData V4
    if "value" in data:
        return data["value"]
    raise ValueError("Unsupported OData response format.")
# ============================================================
# 1. CURRENT STOCK
#
# Material + Plant ONLY
# All storage locations included.
# ============================================================
def get_current_stock(material, plant):
    material = escape_odata(material)
    plant = escape_odata(plant)
    records = odata_get(
        STOCK_SERVICE,
        "A_MatlStkInAcctMod",
        {
            "$filter": f"Material eq '{material}' " f"and Plant eq '{plant}'",
            "$format": "json",
        },
    )
    total = 0.0
    details = []
    for record in records:
        quantity = safe_float(record.get("MatlWrhsStkQtyInMatlBaseUnit"))
        total += quantity
        details.append(
            {
                "StorageLocation": record.get("StorageLocation", ""),
                "Quantity": quantity,
                "Unit": record.get("MaterialBaseUnit", ""),
            }
        )
    return total, details
# ============================================================
# SAP DATE PARSER
# ============================================================
def parse_sap_date(value):
    if not value:
        return None
    value = str(value)
    match = re.search(r"/Date\((-?\d+)", value)
    if match:
        milliseconds = int(match.group(1))
        return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None
# ============================================================
# 2. MATERIAL MOVEMENTS
# ============================================================
def get_material_movements(material, plant):
    material = escape_odata(material)
    plant = escape_odata(plant)
    return odata_get(
        MATERIAL_DOCUMENT_SERVICE,
        "A_MaterialDocumentItem",
        {
            "$filter": f"Material eq '{material}' " f"and Plant eq '{plant}'",
            "$format": "json",
        },
    )
# ============================================================
# CONSUMPTION MOVEMENTS
# ============================================================
CONSUMPTION_MOVEMENT_TYPES = {"201", "221", "261", "281", "601"}
# ============================================================
# MATERIAL DOCUMENT HEADER
# ============================================================
def get_material_document_header(year, document):
    entity = (
        "A_MaterialDocumentHeader"
        f"(MaterialDocumentYear='{year}',"
        f"MaterialDocument='{document}')"
    )
    result = odata_get(MATERIAL_DOCUMENT_SERVICE, entity, {"$format": "json"})
    if not result:
        return None
    return result[0]
# ============================================================
# CONSUMPTION FOR LAST 30 DAYS
# ============================================================
def get_consumption_30d(material, plant):
    movements = get_material_movements(material, plant)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)
    total = 0.0
    details = []
    header_cache = {}
    for movement in movements:
        movement_type = str(movement.get("GoodsMovementType", ""))
        # Only consumption movements
        if movement_type not in CONSUMPTION_MOVEMENT_TYPES:
            continue
        # Ignore cancelled movement
        if movement.get("GoodsMovementIsCancelled") is True:
            continue
        document = str(movement.get("MaterialDocument", ""))
        year = str(movement.get("MaterialDocumentYear", ""))
        if not document or not year:
            continue
        key = (year, document)
        if key not in header_cache:
            header_cache[key] = get_material_document_header(year, document)
        header = header_cache[key]
        if not header:
            continue
        posting_date = parse_sap_date(header.get("PostingDate"))
        if posting_date is None:
            continue
        if not (start_date <= posting_date <= end_date):
            continue
        quantity = abs(safe_float(movement.get("QuantityInBaseUnit")))
        total += quantity
        details.append(
            {
                "MaterialDocument": document,
                "MovementType": movement_type,
                "PostingDate": posting_date.isoformat(),
                "StorageLocation": movement.get("StorageLocation", ""),
                "Quantity": quantity,
            }
        )
    return total, details
# ============================================================
# 3. OPEN PURCHASE ORDERS
#
# Material + Plant only.
# All storage locations.
# ============================================================
def get_open_purchase_orders(material, plant):
    material = escape_odata(material)
    plant = escape_odata(plant)
    records = odata_get(
        PURCHASE_ORDER_SERVICE,
        "A_PurchaseOrderItem",
        {
            "$filter": f"Material eq '{material}' " f"and Plant eq '{plant}'",
            "$expand": "to_PurchaseOrder",
            "$format": "json",
        },
    )
    total = 0.0
    details = []
    for item in records:
        # Deleted
        if item.get("PurchasingDocumentDeletionCode"):
            continue
        # Already completely delivered
        if item.get("IsCompletelyDelivered") is True:
            continue
        # GR not expected
        if item.get("GoodsReceiptIsExpected") is not True:
            continue
        # Return item
        if item.get("IsReturnsItem") is True:
            continue
        quantity = safe_float(item.get("OrderQuantity"))
        total += quantity
        header = item.get("to_PurchaseOrder", {})
        details.append(
            {
                "PurchaseOrder": item.get("PurchaseOrder"),
                "PurchaseOrderItem": item.get("PurchaseOrderItem"),
                "StorageLocation": item.get("StorageLocation", ""),
                "Quantity": quantity,
                "Unit": item.get("PurchaseOrderQuantityUnit", ""),
                "Supplier": header.get("Supplier", ""),
            }
        )
    return total, details
# ============================================================
# 4. SALES ORDER DEMAND
# ============================================================
def get_sales_demand(material, plant):
    material = escape_odata(material)
    plant = escape_odata(plant)
 
    records = odata_get(
        SALES_ORDER_SERVICE,
        "A_SalesOrderItem",
        {
            "$filter":
                f"Material eq '{material}' "
                f"and ProductionPlant eq '{plant}'",
            "$format": "json"
        }
    )

    total_demand = 0.0
    details = []

    for item in records:
        # Ignore rejected sales order items
        if item.get("SalesDocumentRjcnReason"):
            continue
        # Ignore completely processed/delivered items
        if (
            item.get("SDProcessStatus") == "C"
            or item.get("DeliveryStatus") == "C"
        ):
            continue
        requested_qty = safe_float(
            item.get("RequestedQuantity")
        )

        # Active requested quantity is demand
        demand_qty = requested_qty
        if demand_qty <= 0:
            continue
        total_demand += demand_qty

        details.append({
            "SalesOrder": item.get("SalesOrder"),
            "SalesOrderItem": item.get("SalesOrderItem"),
            "Material": item.get("Material"),
            "Plant": item.get("ProductionPlant"),
            "RequestedQuantity": requested_qty,
            "ConfirmedQuantity": safe_float(
                item.get("ConfdDelivQtyInOrderQtyUnit")
            ),
            "DemandQuantity": demand_qty,
            "Unit": item.get("RequestedQuantityUnit"),
            "SDProcessStatus": item.get("SDProcessStatus"),
            "DeliveryStatus": item.get("DeliveryStatus")
        })
 
    return total_demand, details

# ============================================================
# COLLECT CURRENT FEATURES
# ============================================================
def collect_current_features(material, plant):
    current_stock, stock_details = get_current_stock(material, plant)
    consumption, consumption_details = get_consumption_30d(material, plant)
    incoming_po, po_details = get_open_purchase_orders(material, plant)
    demand, demand_details = get_sales_demand(material, plant)
    features = {
        "CurrentStock": current_stock,
        "Consumption30D": consumption,
        "IncomingPO": incoming_po,
        "Demand": demand,
    }
    details = {
        "Stock": stock_details,
        "Consumption": consumption_details,
        "OpenPO": po_details,
        "Demand": demand_details,
    }
    return features, details
# ============================================================
# HISTORICAL DATA
#
# IMPORTANT:
# These rows are ONLY temporary for pipeline testing.
#
# Replace this function with real SAP historical data.
# ============================================================
def build_historical_dataset():
    rows = [
        [2000, 600, 500, 300, 1800],
        [1800, 700, 400, 400, 1600],
        [1600, 800, 300, 500, 1400],
        [1400, 900, 400, 600, 1200],
        [1200, 1000, 300, 700, 900],
        [1000, 1100, 200, 800, 700],
        [800, 1200, 200, 900, 500],
        [600, 1300, 100, 1000, 300],
        [400, 1400, 50, 1100, 150],
        [200, 1500, 0, 1200, 0],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "CurrentStock",
            "Consumption30D",
            "IncomingPO",
            "Demand",
            "StockAfter7Days",
        ],
    )
# ============================================================
# GET SAP AI CORE ACCESS TOKEN
# ============================================================
def get_aicore_access_token():
    required = [AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET]
    if not all(required):
        raise ValueError("SAP AI Core credentials are missing.")
    response = requests.post(
        f"{AICORE_AUTH_URL.rstrip('/')}/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(AICORE_CLIENT_ID, AICORE_CLIENT_SECRET),
        timeout=60,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise ValueError("Access token missing.")
    return token
# ============================================================
# GET TABPFN DEPLOYMENT URL
# ============================================================
def get_tabpfn_deployment_url(access_token):
    if not AICORE_API_URL:
        raise ValueError("AICORE_API_URL missing.")
    if not TABPFN_DEPLOYMENT_ID:
        raise ValueError("TABPFN_DEPLOYMENT_ID missing.")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
    }
    url = f"{AICORE_API_URL.rstrip('/')}" f"/lm/deployments/" f"{TABPFN_DEPLOYMENT_ID}"
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    deployment_url = response.json().get("deploymentUrl")
    if not deployment_url:
        raise ValueError("TabPFN deployment URL missing. " "Check deployment status.")
    return deployment_url
# ============================================================
# CALL DEPLOYED TABPFN MODEL
# ============================================================
def call_tabpfn(X_train, y_train, X_test):
    # --------------------------------------------------------
    # Same deployment concept as your Titanic code.
    # --------------------------------------------------------
    payload = {
        "task_config": {
            "task": "regression",
            "tabpfn_config": {"n_estimators": 8, "random_state": 0},
            "predict_params": {"output_type": "median"},
        },
        "x_train": X_train.to_numpy(dtype=float).tolist(),
        "y_train": y_train.to_numpy(dtype=float).tolist(),
        "x_test": X_test.to_numpy(dtype=float).tolist(),
    }
    # --------------------------------------------------------
    # Authenticate
    # --------------------------------------------------------
    access_token = get_aicore_access_token()
    # --------------------------------------------------------
    # Deployment ID -> Deployment URL
    # --------------------------------------------------------
    deployment_url = get_tabpfn_deployment_url(access_token)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
        "Content-Type": "application/json",
    }
    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    response = requests.post(
        f"{deployment_url.rstrip('/')}/predict",
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    output = response.json()
    if "prediction" not in output:
        raise ValueError("TabPFN response does not " "contain 'prediction'.")
    predictions = np.asarray(output["prediction"], dtype=float).reshape(-1)
    if len(predictions) == 0:
        raise ValueError("TabPFN returned no prediction.")
    if not np.isfinite(predictions).all():
        raise ValueError("TabPFN returned invalid prediction.")
    return predictions
# ============================================================
# HEALTH ENDPOINT
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TabPFN",
        "mode": "SAP AI Core Deployment",
        "deploymentConfigured": bool(TABPFN_DEPLOYMENT_ID),
    }
# ============================================================
# CURRENT FEATURES ENDPOINT
# ============================================================
@app.post("/current-features")
def current_features(request: PredictionRequest):
    try:
        features, details = collect_current_features(request.material, request.plant)
        return {
            "Material": request.material,
            "Plant": request.plant,
            "Features": features,
            "Details": details,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
# ============================================================
# STOCK PREDICTION
# ============================================================
@app.post("/predict-stock-7d")
def predict_stock_7d(request: PredictionRequest):
    try:
        # ----------------------------------------------------
        # STEP 1
        # Get current data from S/4HANA
        # ----------------------------------------------------
        current_features, details = collect_current_features(
            request.material, request.plant
        )
        # ----------------------------------------------------
        # STEP 2
        # Historical dataset
        # ----------------------------------------------------
        history = build_historical_dataset()
        # ----------------------------------------------------
        # STEP 3
        # Prepare TabPFN training/context data
        # ----------------------------------------------------
        X_train = history[FEATURES].astype(float)
        y_train = history[TARGET].astype(float)
        # ----------------------------------------------------
        # STEP 4
        # Current row becomes x_test
        # ----------------------------------------------------
        X_test = pd.DataFrame([current_features], columns=FEATURES).astype(float)
        # ----------------------------------------------------
        # STEP 5
        # Call deployed TabPFN
        # ----------------------------------------------------
        predictions = call_tabpfn(X_train, y_train, X_test)
        predicted_stock = max(0.0, float(predictions[0]))
        # ----------------------------------------------------
        # STEP 6
        # Response
        # ----------------------------------------------------
        return {
            "status": "success",
            "model": "TabPFN",
            "modelSource": "SAP AI Core Deployment",
            "Material": request.material,
            "Plant": request.plant,
            "Input": {
                "CurrentStock": current_features["CurrentStock"],
                "Consumption30D": current_features["Consumption30D"],
                "IncomingPO": current_features["IncomingPO"],
                "Demand": current_features["Demand"],
            },
            "Prediction": {
                "HorizonDays": 7,
                "PredictedStock": round(predicted_stock, 2),
            },
            "Details": details,
        }
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504, detail=("SAP or TabPFN request timed out.")
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=("SAP/AI Core request failed: " f"{str(exc)}")
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
# MAIN
# ============================================================
def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
if __name__ == "__main__":
    main()