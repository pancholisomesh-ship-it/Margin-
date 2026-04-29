from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ml_model import predict_margin
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import logging, traceback, os
import pandas as pd
from pathlib import Path
import json
from flask import Flask
from datetime import datetime
from flask import request, redirect, render_template

app = Flask(__name__)

# 🔥 AUTO RELOAD SETTINGS
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

EXCEL_FILE = Path("data.xlsx")

def save_to_excel(data, margin, biz_type):
    record = data.copy()
    record["business_type"] = biz_type
    record["predicted_margin"] = margin
    df_new = pd.DataFrame([record])
    if EXCEL_FILE.exists():
        df_old = pd.read_excel(EXCEL_FILE)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_excel(EXCEL_FILE, index=False)

load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "business_management")

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client[DB_NAME]
    predictions_collection = db["predictions"]
    solar_data_collection = db["solar_data"]
    paper_data_collection = db["paper_data"]
    mongo_client.server_info()
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.warning(f"MongoDB connection failed: {e}")
    mongo_client = db = predictions_collection = solar_data_collection = paper_data_collection = None

NUMERIC_FIELDS = ["Quantity", "Unit_Price", "Total_Value"]
SOLAR_FIELDS = ["Item_Name", "Brand", "Specification", "Unit", "Quantity", "Unit_Price", "Total_Value", "Supplier", "Purchase_Date"]
PAPER_FIELDS = ["Item_ID", "Bag_Type", "Size", "Material", "GSM", "Color", "Quantity", "Unit_Price", "Total_Value", "Supplier", "Purchase_Date"]

# ===================== CHART DATA GENERATOR =====================
def to_float(val, default=0):
    """Safely convert value to float, handling None."""
    if val is None:
        return float(default)
    return float(val)

def generate_chart_data(biz_type, input_data, margin):
    if biz_type == "solar":
        size = to_float(input_data.get("requirement_kw"), 0)
        plates = to_float(input_data.get("plates_cost"), 0)
        inverter = to_float(input_data.get("inverter_cost"), 0)
        install = to_float(input_data.get("install_cost"), 0)
        selling = to_float(input_data.get("selling_price"), 0)
        total_cost = plates + inverter + install

        return {
            "margin_pie": {
                "labels": ["Gross Margin", "Cost Portion"],
                "values": [margin, 100 - margin]
            },
            "cost_breakdown": {
                "labels": ["Plates", "Inverter", "Installation"],
                "values": [plates, inverter, install]
            },
            "profit_analysis": {
                "revenue": selling,
                "total_cost": total_cost,
                "profit": selling - total_cost
            }
        }
    else:  # paper
        qty = to_float(input_data.get("quantity"), 0)
        sell_price = to_float(input_data.get("selling_price"), 0)
        raw = to_float(input_data.get("raw_cost"), 0)
        labor = to_float(input_data.get("labor_cost"), 0)
        elec = to_float(input_data.get("elec_cost"), 0)
        pack = to_float(input_data.get("pack_cost"), 0)

        total_revenue = qty * sell_price
        total_cost = (raw + labor + elec + pack) * qty

        return {
            "margin_pie": {
                "labels": ["Gross Margin", "Cost Portion"],
                "values": [margin, 100 - margin]
            },
            "cost_breakdown": {
                "labels": ["Raw Material", "Labor", "Electricity", "Packaging"],
                "values": [raw*qty, labor*qty, elec*qty, pack*qty]
            },
            "profit_analysis": {
                "revenue": total_revenue,
                "total_cost": total_cost,
                "profit": total_revenue - total_cost
            }
        }

# ===================== UPDATED PREDICT ROUTE =====================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        biz_type = data.get("type")

        if biz_type == "solar":
            # Updated Solar fields (removed labour_cost from validation)
            solar_fields = ["requirement_kw", "plates_cost", "inverter_cost", "install_cost", "selling_price"]
            errors = validate_fields(data, solar_fields)
            collection = solar_data_collection
        elif biz_type == "paper":
            errors = validate_fields(data, PAPER_FIELDS)
            collection = paper_data_collection
        else:
            return jsonify({"error": "Use solar or paper"}), 400

        if errors:
            return jsonify({"error": "Validation Failed", "details": errors}), 422

        features = extract_features(data)
        margin = round(float(predict_margin(features)), 4)

        # Generate chart data
        chart_data = generate_chart_data(biz_type, data, margin)

        # Save to MongoDB
        predictions_collection.insert_one({
            "timestamp": datetime.utcnow(),
            "business_type": biz_type,
            "input_data": data,
            "predicted_margin": margin
        })

        if collection is not None:
            rec = data.copy()
            rec["created_at"] = datetime.utcnow()
            rec["predicted_margin"] = margin
            collection.insert_one(rec)

        save_to_excel(data, margin, biz_type)

        return jsonify({
            "margin": margin,
            "business_type": biz_type,
            "features_used": features,
            "chart_data": chart_data
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500


# ===================== NEW BULK CSV PREDICTION ROUTE =====================
@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        biz_type = request.form.get("type")

        if biz_type not in ["solar", "paper"]:
            return jsonify({"error": "Invalid business type"}), 400

        df = pd.read_csv(file)

        results = []
        for _, row in df.iterrows():
            data = row.to_dict()

            if biz_type == "solar":
                # Clean column names for solar
                data['requirement_kw'] = data.get('requirement_kw') or data.get('size_kw')
                data['plates_cost'] = data.get('plates_cost')
                data['inverter_cost'] = data.get('inverter_cost')
                data['install_cost'] = data.get('install_cost')
                data['selling_price'] = data.get('selling_price')

            features = extract_features(data)
            margin = round(float(predict_margin(features)), 4)
            chart_data = generate_chart_data(biz_type, data, margin)

            results.append({
                "input": data,
                "margin": margin,
                "chart_data": chart_data
            })

        return jsonify({
            "business_type": biz_type,
            "total_predictions": len(results),
            "results": results
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
def validate_fields(data, required_fields):
    errors = []
    for field in required_fields:
        value = data.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"{field} is required")
            continue
        if field in NUMERIC_FIELDS:
            try:
                float(value)
            except:
                errors.append(f"{field} must be numeric")
    return errors

def extract_features(data):
    type_code = 1 if data.get("type") == "solar" else 0
    return [type_code, float(data.get("Quantity",0)), float(data.get("Unit_Price",0)), float(data.get("Total_Value",0))]

# =====================================================
# WEBSITE ROUTES
# =====================================================

@app.route("/")
def login():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/eco_analytics")
def eco():
    return render_template("eco.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/prediction")
def prediction():
    return render_template("solar_paperbags.html")

@app.route("/health")
def health():
    return jsonify({"status":"running","mongodb_connected": mongo_client is not None})

@app.route("/excel-data")
def excel_data():
    try:
        df = pd.read_excel("data.xlsx")

        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/profit-data")
def profit_data():
    total = predictions_collection.count_documents({})
    return jsonify({"total_profit": total * 3800})

@app.route("/orders-data")
def orders_data():
    orders = paper_data_collection.count_documents({})
    return jsonify({"orders": orders})

@app.route("/model-info")
def model_info():
    return jsonify({"version": "2.1"})

@app.route("/inventory-data")
def inventory_data():
    return jsonify({"waste": "-18%"})

@app.route("/leads")
def leads():

    enquiries = db.enquiries.find().sort("time",-1)

    return render_template(
        "leads.html",
        enquiries=enquiries
    )
    
@app.route("/submit_enquiry", methods=["POST"])
def submit_enquiry():
    enquiry = {
        "name": request.form["first_name"] + " " + request.form["last_name"],
        "email": request.form["email"],
        "phone": request.form["phone"],
        "enquiry_type": request.form["enquiry_type"],
        "company": request.form["company"],
        "message": request.form["message"],
        "status": "New Lead",
        "time": datetime.now()
    }

    db.enquiries.insert_one(enquiry)

    return redirect("/contact")
# =====================================================
# MONGODB DATA API
# =====================================================

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    if predictions_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        limit = request.args.get("limit", 50, type=int)
        predictions = list(predictions_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        return jsonify({"predictions": predictions, "count": len(predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/solar-data", methods=["GET"])
def get_solar_data():
    if solar_data_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        limit = request.args.get("limit", 50, type=int)
        data = list(solar_data_collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
        return jsonify({"data": data, "count": len(data)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/paper-data", methods=["GET"])
def get_paper_data():
    if paper_data_collection is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        limit = request.args.get("limit", 50, type=int)
        data = list(paper_data_collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
        return jsonify({"data": data, "count": len(data)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def get_stats():
    if db is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    try:
        stats = {
            "predictions_count": predictions_collection.count_documents({}),
            "solar_records_count": solar_data_collection.count_documents({}),
            "paper_records_count": paper_data_collection.count_documents({})
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# PREDICTION API
# =====================================================


# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

# =====================================================
# RUN SERVER
# =====================================================

# app = Flask(__name__)
# app.config['TEMPLATES_AUTO_RELOAD'] = True

# if __name__ == "__main__":
#     app.run(debug=True)
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)