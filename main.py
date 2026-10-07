import os
import re
from datetime import datetime, timedelta, timezone
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
app = FastAPI(
    title="Supply Chain Stock Prediction API",
    version="2.0.0"
)

# ============================================================
# S/4HANA CONFIGURATION
# ============================================================
S4_BASE_URL = os.getenv(
    "S4_BASE_URL",
    "http://192.168.8.69:50000"
).rstrip("/")
S4_USERNAME = os.getenv("S4_USERNAME")
S4_PASSWORD = os.getenv("S4_PASSWORD")
VERIFY_SSL = (
    os.getenv("VERIFY_SSL", "false").lower() == "true"
)

# ============================================================
# SAP AI CORE / TABPFN CONFIGURATION
# ============================================================
AICORE_AUTH_URL = os.getenv("AICORE_AUTH_URL")
AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID")
AICORE_CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET")
AICORE_API_URL = os.getenv("AICORE_API_URL")
TABPFN_DEPLOYMENT_ID = os.getenv("TABPFN_DEPLOYMENT_ID")
AICORE_RESOURCE_GROUP = os.getenv(
    "AICORE_RESOURCE_GROUP",
    "default"
)

# ============================================================
# S/4HANA SERVICE PATHS
# ============================================================
STOCK_SERVICE = "/sap/opu/odata/sap/API_MATERIAL_STOCK_SRV"
MATERIAL_DOCUMENT_SERVICE = (
    "/sap/opu/odata/sap/API_MATERIAL_DOCUMENT_SRV"
)
PURCHASE_ORDER_SERVICE = (
    "/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV"
)
SALES_ORDER_SERVICE = "/sap/opu/odata/sap/API_SALES_ORDER_SRV"

# ============================================================
# TABPFN FEATURES
# ============================================================
FEATURES = [
    "CurrentStock",
    "Consumption30D",
    "IncomingPO",
    "Demand"
]
TARGET = "StockAfter7Days"

# ============================================================
# REQUEST MODEL
# ============================================================
class PredictionRequest(BaseModel):
    material: str
    plant: str

# ============================================================
# S/4 HTTP SESSION
# ============================================================
s4_session = requests.Session()
s4_session.auth = HTTPBasicAuth(
    S4_USERNAME,
    S4_PASSWORD
)
s4_session.headers.update({
    "Accept": "application/json"
})

# ============================================================
# GENERAL HELPERS
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
# SAP DATE PARSER
# Supports:
# /Date(1790640000000)/
# and ISO timestamps.
# ============================================================
def parse_sap_date(value):
    """
    Parse SAP OData V2 /Date(...)/ or ISO dates.
    Missing/invalid dates return None so one incomplete SAP row
    does not stop the entire stock prediction.
    """
    if value is None or str(value).strip() == "":
        return None

    value = str(value).strip()
    match = re.search(
        r"/Date\((-?\d+)(?:[+-]\d+)?\)/",
        value
    )

    if match:
        try:
            milliseconds = int(match.group(1))
            return datetime.fromtimestamp(
                milliseconds / 1000,
                tz=timezone.utc
            )
        except (ValueError, OverflowError, OSError):
            return None

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ============================================================
# GENERIC ODATA GET
# ============================================================
def odata_get(service, entity, params=None):
    url = (
        f"{S4_BASE_URL}"
        f"{service}/"
        f"{entity}"
    )
    response = s4_session.get(
        url,
        params=params,
        verify=VERIFY_SSL,
        timeout=120
    )
    # Better error visibility
    if not response.ok:
        raise ValueError(
            "\nSAP OData request failed\n"
            f"Status: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"Response: {response.text}"
        )
    data = response.json()
    # -----------------------------
    # OData V2
    # -----------------------------
    if "d" in data:
        result = data["d"]
        if isinstance(result, dict):
            if "results" in result:
                return result["results"]
            return [result]
    # -----------------------------
    # OData V4
    # -----------------------------
    if "value" in data:
        return data["value"]
    raise ValueError(
        "Unsupported OData response format."
    )

# ============================================================
# 1. CURRENT STOCK
# Material + Plant
# All storage locations are included.
# ============================================================
def get_current_stock(material, plant):
    material_filter = escape_odata(material)
    plant_filter = escape_odata(plant)
    records = odata_get(
        STOCK_SERVICE,
        "A_MatlStkInAcctMod",
        {
            "$filter":
                f"Material eq '{material_filter}' "
                f"and Plant eq '{plant_filter}'",
            "$format": "json"
        }
    )
    total = 0.0
    details = []
    for record in records:
        quantity = safe_float(
            record.get("MatlWrhsStkQtyInMatlBaseUnit")
        )
        total += quantity
        details.append({
            "StorageLocation":
                record.get("StorageLocation", ""),
            "Quantity": quantity,
            "Unit":
                record.get("MaterialBaseUnit", "")
        })
    return total, details

# ============================================================
# 2. MATERIAL MOVEMENTS
# IMPORTANT:
# Expand Material Document Header so PostingDate is available.
# ============================================================
def get_material_movements(material, plant):
    material_filter = escape_odata(material)
    plant_filter = escape_odata(plant)
    return odata_get(
        MATERIAL_DOCUMENT_SERVICE,
        "A_MaterialDocumentItem",
        {
            "$filter":
                f"Material eq '{material_filter}' "
                f"and Plant eq '{plant_filter}'",
            "$expand":
                "to_MaterialDocumentHeader",
            "$format": "json"
        }
    )

# ============================================================
# CONSUMPTION MOVEMENT TYPES
# ============================================================
CONSUMPTION_MOVEMENT_TYPES = {
    # Goods issue to cost center
    "201",
    # Goods issue to project
    "221",
    # Goods issue to production order
    "261",
    # Goods issue to network
    "281",
    # Goods issue for delivery
    "601"
}

# ============================================================
# 2A. CONSUMPTION LAST 30 DAYS
# Window end:
#   latest (default) = latest consumption posting date
#                      (for historical demo data)
#   today            = current date (for live data)
# Set CONSUMPTION_WINDOW_END=today in .env for live data.
# ============================================================
def get_consumption_30d(material, plant):
    movements = get_material_movements(
        material,
        plant
    )
    # Normalise types so "0261", " 261" and "261" all match
    consumption_types = {
        str(t).strip().lstrip("0")
        for t in CONSUMPTION_MOVEMENT_TYPES
    }
    skipped = {
        "not_consumption_type": 0,
        "cancelled": 0,
        "no_posting_date": 0,
        "invalid_posting_date": 0,
        "zero_quantity": 0
    }
    print("\n====================================")
    print("CONSUMPTION DEBUG")
    print("====================================")
    print("Material:", material, "| Plant:", plant)
    print("Total movements:", len(movements))
    print(
        "Movement types found:",
        sorted({
            str(m.get("GoodsMovementType", "")).strip()
            for m in movements
        })
    )
    header_cache = {}
    records = []
    for movement in movements:
        # ------------------------------------------
        # Movement Type
        # ------------------------------------------
        movement_type = str(
            movement.get("GoodsMovementType", "")
        ).strip()
        if movement_type.lstrip("0") not in consumption_types:
            skipped["not_consumption_type"] += 1
            continue
        # ------------------------------------------
        # Ignore cancelled documents
        # ------------------------------------------
        if str(
            movement.get("GoodsMovementIsCancelled", "")
        ).strip().lower() == "true":
            skipped["cancelled"] += 1
            continue
        # ------------------------------------------
        # Posting Date
        # 1) item  2) expanded header  3) header call
        # ------------------------------------------
        posting_date_raw = movement.get("PostingDate")
        if not posting_date_raw:
            header = movement.get(
                "to_MaterialDocumentHeader"
            ) or {}
            if isinstance(header, dict) and "results" in header:
                header_results = header.get("results") or []
                header = header_results[0] if header_results else {}
            posting_date_raw = header.get("PostingDate")
        if not posting_date_raw:
            year = str(
                movement.get("MaterialDocumentYear", "")
            ).strip()
            document = str(
                movement.get("MaterialDocument", "")
            ).strip()
            item_no = str(
                movement.get("MaterialDocumentItem", "")
            ).strip()
            cache_key = (year, document)
            if year and document and item_no:
                if cache_key not in header_cache:
                    try:
                        entity = (
                            "A_MaterialDocumentItem("
                            f"MaterialDocumentYear='{escape_odata(year)}',"
                            f"MaterialDocument='{escape_odata(document)}',"
                            f"MaterialDocumentItem='{escape_odata(item_no)}'"
                            ")/to_MaterialDocumentHeader"
                        )
                        result = odata_get(
                            MATERIAL_DOCUMENT_SERVICE,
                            entity,
                            {"$format": "json"}
                        )
                        header_cache[cache_key] = (
                            result[0] if result else {}
                        )
                    except Exception as exc:
                        print(
                            "HEADER ERROR:",
                            year, document, item_no, str(exc)
                        )
                        header_cache[cache_key] = {}
                posting_date_raw = header_cache[cache_key].get(
                    "PostingDate"
                )
        if not posting_date_raw:
            skipped["no_posting_date"] += 1
            continue
        posting_date = parse_sap_date(posting_date_raw)
        if posting_date is None:
            skipped["invalid_posting_date"] += 1
            print("INVALID POSTING DATE:", posting_date_raw)
            continue
        # ------------------------------------------
        # Quantity
        # ------------------------------------------
        quantity = abs(
            safe_float(movement.get("QuantityInBaseUnit"))
        )
        if quantity == 0:
            skipped["zero_quantity"] += 1
            continue
        records.append({
            "MaterialDocument":
                movement.get("MaterialDocument"),
            "MaterialDocumentYear":
                movement.get("MaterialDocumentYear"),
            "MaterialDocumentItem":
                movement.get("MaterialDocumentItem"),
            "MovementType":
                movement_type,
            "PostingDateObject":
                posting_date,
            "PostingDate":
                posting_date.isoformat(),
            "StorageLocation":
                movement.get("StorageLocation", ""),
            "Quantity":
                quantity,
            "Unit":
                movement.get("MaterialBaseUnit", "")
        })
    print("Skipped summary:", skipped)
    print("Valid consumption movements:", len(records))
    if not records:
        print("Consumption30D = 0 (no valid consumption movements)")
        print("====================================")
        return 0.0, []
    # ----------------------------------------------
    # Window end
    # ----------------------------------------------
    window_mode = os.getenv(
        "CONSUMPTION_WINDOW_END",
        "latest"
    ).strip().lower()
    if window_mode == "today":
        end_date = datetime.now(timezone.utc)
    else:
        end_date = max(
            r["PostingDateObject"] for r in records
        )
    start_date = end_date - timedelta(days=30)
    print("Window mode:", window_mode)
    print("Window:", start_date.date(), "->", end_date.date())
    # ----------------------------------------------
    # Sum last 30 days
    # ----------------------------------------------
    total = 0.0
    details = []
    for record in records:
        posting_date = record["PostingDateObject"]
        if posting_date < start_date or posting_date > end_date:
            continue
        total += record["Quantity"]
        details.append({
            key: value
            for key, value in record.items()
            if key != "PostingDateObject"
        })
    print("Consumption30D:", total)
    print("Transactions counted:", len(details))
    print("====================================")
    return round(total, 3), details

# ============================================================
# 3. OPEN PURCHASE ORDERS
# CURRENT IMPLEMENTATION:
# Material + Plant
# Includes all storage locations.
# NOTE:
# This currently uses OrderQuantity.
# Later we can improve this to:
# IncomingPO7D using PO Schedule Lines.
# ============================================================
def get_open_purchase_orders(material, plant):
    material_filter = escape_odata(material)
    plant_filter = escape_odata(plant)
    records = odata_get(
        PURCHASE_ORDER_SERVICE,
        "A_PurchaseOrderItem",
        {
            "$filter":
                f"Material eq '{material_filter}' "
                f"and Plant eq '{plant_filter}'",
            "$expand":
                "to_PurchaseOrder",
            "$format": "json"
        }
    )
    total = 0.0
    details = []
    for item in records:
        # Deleted PO item
        if item.get("PurchasingDocumentDeletionCode"):
            continue
        # Completely delivered
        if item.get("IsCompletelyDelivered") is True:
            continue
        # Goods receipt not expected
        if item.get("GoodsReceiptIsExpected") is not True:
            continue
        # Return PO
        if item.get("IsReturnsItem") is True:
            continue
        quantity = safe_float(item.get("OrderQuantity"))
        if quantity <= 0:
            continue
        total += quantity
        header = item.get("to_PurchaseOrder") or {}
        # Handle expanded relationship
        if isinstance(header, dict) and "results" in header:
            header_results = header.get("results") or []
            header = (
                header_results[0]
                if header_results
                else {}
            )
        details.append({
            "PurchaseOrder":
                item.get("PurchaseOrder"),
            "PurchaseOrderItem":
                item.get("PurchaseOrderItem"),
            "StorageLocation":
                item.get("StorageLocation", ""),
            "Quantity":
                quantity,
            "Unit":
                item.get("PurchaseOrderQuantityUnit", ""),
            "Supplier":
                header.get("Supplier", "")
        })
    return total, details

# ============================================================
# 4. SALES ORDER DEMAND
# IMPORTANT:
# Sales Order API uses:
# ProductionPlant
# NOT:
# Plant
# ============================================================
def get_sales_demand(material, plant):
    material_filter = escape_odata(material)
    plant_filter = escape_odata(plant)
    records = odata_get(
        SALES_ORDER_SERVICE,
        "A_SalesOrderItem",
        {
            "$filter":
                f"Material eq '{material_filter}' "
                f"and ProductionPlant eq "
                f"'{plant_filter}'",
            "$format": "json"
        }
    )
    total_demand = 0.0
    details = []
    for item in records:
        # Ignore rejected items
        if item.get("SalesDocumentRjcnReason"):
            continue
        # Ignore completed items
        if (
            item.get("SDProcessStatus") == "C"
            or item.get("DeliveryStatus") == "C"
        ):
            continue
        requested_qty = safe_float(
            item.get("RequestedQuantity")
        )
        if requested_qty <= 0:
            continue
        # ------------------------------------------
        # Current demand logic
        # IMPORTANT:
        # Do NOT use:
        # requested - confirmed
        # because confirmed quantity can still
        # represent future customer requirement.
        # ------------------------------------------
        demand_qty = requested_qty
        total_demand += demand_qty
        details.append({
            "SalesOrder":
                item.get("SalesOrder"),
            "SalesOrderItem":
                item.get("SalesOrderItem"),
            "Material":
                item.get("Material"),
            "Plant":
                item.get("ProductionPlant"),
            "RequestedQuantity":
                requested_qty,
            "ConfirmedQuantity":
                safe_float(
                    item.get("ConfdDelivQtyInOrderQtyUnit")
                ),
            "DemandQuantity":
                demand_qty,
            "Unit":
                item.get("RequestedQuantityUnit"),
            "SDProcessStatus":
                item.get("SDProcessStatus"),
            "DeliveryStatus":
                item.get("DeliveryStatus")
        })
    return total_demand, details

# ============================================================
# 5. COLLECT CURRENT FEATURES
# ============================================================
def collect_current_features(material, plant):
    # Current Stock
    (
        current_stock,
        stock_details
    ) = get_current_stock(material, plant)
    # Consumption
    (
        consumption,
        consumption_details
    ) = get_consumption_30d(material, plant)
    # Incoming PO
    (
        incoming_po,
        po_details
    ) = get_open_purchase_orders(material, plant)
    # Demand
    (
        demand,
        demand_details
    ) = get_sales_demand(material, plant)
    features = {
        "CurrentStock":
            current_stock,
        "Consumption30D":
            consumption,
        "IncomingPO":
            incoming_po,
        "Demand":
            demand
    }
    details = {
        "Stock":
            stock_details,
        "Consumption":
            consumption_details,
        "OpenPO":
            po_details,
        "Demand":
            demand_details
    }
    return features, details

# ============================================================
# 6. HISTORICAL DATA
# IMPORTANT:
# TEMPORARY TEST DATA ONLY.
# DO NOT consider final TabPFN prediction
# business-valid until this is replaced
# with real historical SAP observations.
# ============================================================
TRAINING_CSV_PATH = os.getenv(
    "TRAINING_CSV_PATH",
    "stockout_training_dummy_1050.csv"
)

def resolve_training_csv_path():
    """Find the training CSV without depending on a numbered upload filename."""
    candidates = [
        TRAINING_CSV_PATH,
        "stockout_training_dummy_1050.csv",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise ValueError(
        "Training CSV not found. Set TRAINING_CSV_PATH in .env or keep "
        "stockout_training_dummy_1050.csv beside app.py."
    )

def build_historical_dataset():
    """
    Load the supplied dummy CSV for TabPFN training.

    CSV columns:
      CurrentStock
      Consumption30D
      IncomingPO7D
      Demand7D
      StockAfter7D

    The old working S/4 code produces:
      CurrentStock
      Consumption30D
      IncomingPO
      Demand

    Therefore IncomingPO7D -> IncomingPO and Demand7D -> Demand are mapped
    before building X_train. CSV Date is not required for prediction.
    """
    csv_path = resolve_training_csv_path()
    history = pd.read_csv(csv_path)

    csv_required = [
        "CurrentStock",
        "Consumption30D",
        "IncomingPO7D",
        "Demand7D",
        "StockAfter7D",
    ]

    missing = [c for c in csv_required if c not in history.columns]
    if missing:
        raise ValueError(
            "Training CSV missing required columns: " + ", ".join(missing)
        )

    # Use only TRAIN rows when DataSplit exists.
    if "DataSplit" in history.columns:
        split = history["DataSplit"].astype(str).str.strip().str.upper()
        train_rows = history[split == "TRAIN"].copy()
        if not train_rows.empty:
            history = train_rows

    # Map CSV feature names to the names used by the old working API.
    history = history.rename(
        columns={
            "IncomingPO7D": "IncomingPO",
            "Demand7D": "Demand",
            "StockAfter7D": "StockAfter7Days",
        }
    )

    required = FEATURES + [TARGET]
    for column in required:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce"
        )

    history = history.replace([np.inf, -np.inf], np.nan)
    history = history.dropna(subset=required).reset_index(drop=True)

    if len(history) < 20:
        raise ValueError(
            f"Training CSV has only {len(history)} usable TRAIN rows."
        )

    print("\n====================================")
    print("CSV TRAINING DATA")
    print("====================================")
    print("CSV:", csv_path)
    print("Training rows:", len(history))
    print("Features:", FEATURES)
    print("Target:", TARGET)
    print("NOTE: CSV Date is not used by TabPFN prediction.")
    print("====================================")

    return history

# ============================================================
# 7. SAP AI CORE ACCESS TOKEN
# ============================================================
def get_aicore_access_token():
    required = [
        AICORE_AUTH_URL,
        AICORE_CLIENT_ID,
        AICORE_CLIENT_SECRET
    ]
    if not all(required):
        raise ValueError(
            "SAP AI Core credentials are missing."
        )
    response = requests.post(
        f"{AICORE_AUTH_URL.rstrip('/')}"
        f"/oauth/token",
        data={
            "grant_type": "client_credentials"
        },
        auth=(
            AICORE_CLIENT_ID,
            AICORE_CLIENT_SECRET
        ),
        timeout=60
    )
    if not response.ok:
        raise ValueError(
            "AI Core authentication failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )
    token = response.json().get("access_token")
    if not token:
        raise ValueError(
            "AI Core access token missing."
        )
    return token

# ============================================================
# 8. GET TABPFN DEPLOYMENT URL
# ============================================================
def get_tabpfn_deployment_url(access_token):
    if not AICORE_API_URL:
        raise ValueError(
            "AICORE_API_URL is missing."
        )
    if not TABPFN_DEPLOYMENT_ID:
        raise ValueError(
            "TABPFN_DEPLOYMENT_ID is missing."
        )
    headers = {
        "Authorization":
            f"Bearer {access_token}",
        "AI-Resource-Group":
            AICORE_RESOURCE_GROUP
    }
    url = (
        f"{AICORE_API_URL.rstrip('/')}"
        f"/lm/deployments/"
        f"{TABPFN_DEPLOYMENT_ID}"
    )
    response = requests.get(
        url,
        headers=headers,
        timeout=60
    )
    if not response.ok:
        raise ValueError(
            "Unable to get TabPFN deployment.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )
    deployment_url = response.json().get("deploymentUrl")
    if not deployment_url:
        raise ValueError(
            "TabPFN deployment URL missing. "
            "Check deployment status."
        )
    return deployment_url

# ============================================================
# 9. CALL DEPLOYED TABPFN
# ============================================================
def call_tabpfn(X_train, y_train, X_test):
    # TabPFN regression payload
    payload = {
        "task_config": {
            "task": "regression",
            "tabpfn_config": {
                "n_estimators": 8,
                "random_state": 0
            },
            "predict_params": {
                "output_type": "median"
            }
        },
        "x_train":
            X_train.to_numpy(dtype=float).tolist(),
        "y_train":
            y_train.to_numpy(dtype=float).tolist(),
        "x_test":
            X_test.to_numpy(dtype=float).tolist()
    }
    # Authenticate
    access_token = get_aicore_access_token()
    # Deployment ID -> URL
    deployment_url = get_tabpfn_deployment_url(
        access_token
    )
    headers = {
        "Authorization":
            f"Bearer {access_token}",
        "AI-Resource-Group":
            AICORE_RESOURCE_GROUP,
        "Content-Type":
            "application/json"
    }
    # Prediction
    response = requests.post(
        f"{deployment_url.rstrip('/')}"
        f"/predict",
        headers=headers,
        json=payload,
        timeout=300
    )
    if not response.ok:
        raise ValueError(
            "TabPFN prediction failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}"
        )
    output = response.json()
    if "prediction" not in output:
        raise ValueError(
            "TabPFN response does not "
            "contain 'prediction'."
        )
    predictions = np.asarray(
        output["prediction"],
        dtype=float
    ).reshape(-1)
    if len(predictions) == 0:
        raise ValueError(
            "TabPFN returned no prediction."
        )
    if not np.isfinite(predictions).all():
        raise ValueError(
            "TabPFN returned invalid prediction."
        )
    return predictions

# ============================================================
# 10. HEALTH ENDPOINT
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "TabPFN",
        "mode": "SAP AI Core Deployment",
        "deploymentConfigured":
            bool(TABPFN_DEPLOYMENT_ID)
    }

# ============================================================
# 11. CURRENT FEATURES ENDPOINT
# TEST THIS FIRST.
# ============================================================
@app.post("/current-features")
def current_features(request: PredictionRequest):
    try:
        (
            features,
            details
        ) = collect_current_features(
            request.material,
            request.plant
        )
        return {
            "Material":
                request.material,
            "Plant":
                request.plant,
            "Features":
                features,
            "Details":
                details
        }
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="SAP request timed out."
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

# ============================================================
# 12. STOCK PREDICTION ENDPOINT
# ============================================================
@app.post("/predict-stock-7d")
def predict_stock_7d(request: PredictionRequest):
    try:
        # STEP 1: Get current S/4HANA data
        (
            current_features,
            details
        ) = collect_current_features(
            request.material,
            request.plant
        )
        # STEP 2: Historical TabPFN context
        history = build_historical_dataset()
        if history.empty:
            raise ValueError(
                "Historical dataset is empty."
            )
        # STEP 3: X train / Y train
        X_train = history[FEATURES].astype(float)
        y_train = history[TARGET].astype(float)
        # STEP 4: Current SAP row becomes X_test
        X_test = pd.DataFrame(
            [current_features],
            columns=FEATURES
        ).astype(float)
        # STEP 5: Call TabPFN on SAP AI Core
        predictions = call_tabpfn(
            X_train,
            y_train,
            X_test
        )
        predicted_stock = max(
            0.0,
            float(predictions[0])
        )
        # STEP 6: Return response
        return {
            "status": "success",
            "model": "TabPFN",
            "modelSource": "SAP AI Core Deployment",
            "Material":
                request.material,
            "Plant":
                request.plant,
            "Input": {
                "CurrentStock":
                    current_features["CurrentStock"],
                "Consumption30D":
                    current_features["Consumption30D"],
                "IncomingPO":
                    current_features["IncomingPO"],
                "Demand":
                    current_features["Demand"]
            },
            "Prediction": {
                "HorizonDays": 7,
                "PredictedStock":
                    round(predicted_stock, 2)
            },
            "Details":
                details,
            # Important reminder while testing
            "PredictionDataStatus":
                "TEMPORARY_HISTORICAL_DATA"
        }
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="SAP or TabPFN request timed out."
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "SAP/AI Core request failed: "
                f"{str(exc)}"
            )
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

# ============================================================
# MAIN
# ============================================================
def main():
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000"))
    )

if __name__ == "__main__":
    main()