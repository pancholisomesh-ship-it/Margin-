from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from datetime import datetime, timedelta
from functools import wraps
from pymongo import MongoClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import random
from ml_model import predict_margin, get_model_info
from business_config import BUSINESS_TYPES, get_business_fields, get_business

# =====================================================
# APP INIT
# =====================================================

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.secret_key = "your_super_secret_key_here_change_in_production"  # ← Change this in production

# =====================================================
# MONGODB — dealers only
# =====================================================

mongodb_available = False
client = None
db = None
dealers_collection = None
support_collection = None
users_collection = None

# Fallback support message storage when MongoDB is unavailable
support_messages = []

try:
    client = MongoClient(
        "mongodb+srv://admin:root@cluster0.cvf0hfq.mongodb.net/?retryWrites=true&w=majority",
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")
    db = client["somesh"]
    dealers_collection = db["dealers"]
    support_collection = db["support_messages"]
    users_collection = db["users"]
    mongodb_available = True
except Exception as e:
    print(f"Warning: MongoDB unavailable: {e}")
    print("MongoDB Atlas is not reachable. Continuing with in-memory fallback for support messages.")


def require_db():
    if not mongodb_available:
        return jsonify({"error": "MongoDB is temporarily unavailable."}), 503
    return None

# =====================================================
# GMAIL CONFIG
# =====================================================
# Steps to get App Password:
#   1. myaccount.google.com → Security → 2-Step Verification → ON
#   2. Security → App Passwords → Generate
#   3. Paste the 16-char password below

GMAIL_USER     = "your_gmail@gmail.com"   # ← Change this
GMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"    # ← Paste App Password here
BASE_URL       = "http://127.0.0.1:5000"  # ← Change to your domain in production

# =====================================================
# MEMORY STORE — pre-seeded with demo data
# =====================================================

prediction_store = {
    "solar": [
        {
            "prediction": 42.5,
            "timestamp": "2026-04-01 10:00",
            "breakdown": [
                {"label": "Total Revenue",    "value": "₹1,50,000", "color": "var(--accent2)"},
                {"label": "Total Cost",       "value": "₹86,250",   "color": "var(--danger)"},
                {"label": "Gross Profit",     "value": "₹63,750",   "color": "var(--accent)"},
                {"label": "Plate Total Cost", "value": "₹60,000",   "color": "var(--muted)"}
            ],
            "chart": {
                "margin_pie": {"labels": ["Gross Margin", "Cost"], "values": [42.5, 57.5]},
                "cost_breakdown": {"labels": ["Inverter", "Installation", "Plates"], "values": [15000, 11250, 60000]}
            }
        },
        {
            "prediction": 38.2,
            "timestamp": "2026-04-10 14:30",
            "breakdown": [
                {"label": "Total Revenue",    "value": "₹2,00,000", "color": "var(--accent2)"},
                {"label": "Total Cost",       "value": "₹1,23,560", "color": "var(--danger)"},
                {"label": "Gross Profit",     "value": "₹76,440",   "color": "var(--accent)"},
                {"label": "Plate Total Cost", "value": "₹90,000",   "color": "var(--muted)"}
            ],
            "chart": {
                "margin_pie": {"labels": ["Gross Margin", "Cost"], "values": [38.2, 61.8]},
                "cost_breakdown": {"labels": ["Inverter", "Installation", "Plates"], "values": [18000, 15560, 90000]}
            }
        },
        {
            "prediction": 51.0,
            "timestamp": "2026-04-20 09:15",
            "breakdown": [
                {"label": "Total Revenue",    "value": "₹2,50,000", "color": "var(--accent2)"},
                {"label": "Total Cost",       "value": "₹1,22,500", "color": "var(--danger)"},
                {"label": "Gross Profit",     "value": "₹1,27,500", "color": "var(--accent)"},
                {"label": "Plate Total Cost", "value": "₹95,000",   "color": "var(--muted)"}
            ],
            "chart": {
                "margin_pie": {"labels": ["Gross Margin", "Cost"], "values": [51.0, 49.0]},
                "cost_breakdown": {"labels": ["Inverter", "Installation", "Plates"], "values": [12000, 15500, 95000]}
            }
        }
    ],
    "paperbags": [
        {
            "prediction": 38.0,
            "timestamp": "2026-04-02 11:30",
            "breakdown": [
                {"label": "Total Revenue",   "value": "₹80,000",  "color": "var(--accent2)"},
                {"label": "Total Cost",      "value": "₹49,600",  "color": "var(--danger)"},
                {"label": "Gross Profit",    "value": "₹30,400",  "color": "var(--accent)"},
                {"label": "Per Unit Margin", "value": "₹30.40",   "color": "var(--warn)"}
            ],
            "chart": {
                "margin_pie": {"labels": ["Gross Margin", "Cost"], "values": [38.0, 62.0]},
                "cost_breakdown": {"labels": ["Raw Material", "Labour", "Delivery"], "values": [32000, 12000, 5600]}
            }
        },
        {
            "prediction": 44.5,
            "timestamp": "2026-04-12 16:00",
            "breakdown": [
                {"label": "Total Revenue",   "value": "₹1,20,000", "color": "var(--accent2)"},
                {"label": "Total Cost",      "value": "₹66,600",   "color": "var(--danger)"},
                {"label": "Gross Profit",    "value": "₹53,400",   "color": "var(--accent)"},
                {"label": "Per Unit Margin", "value": "₹53.40",    "color": "var(--warn)"}
            ],
            "chart": {
                "margin_pie": {"labels": ["Gross Margin", "Cost"], "values": [44.5, 55.5]},
                "cost_breakdown": {"labels": ["Raw Material", "Labour", "Delivery"], "values": [45000, 14000, 7600]}
            }
        },
        {
            "prediction": 35.8,
            "timestamp": "2026-04-22 13:45",
            "breakdown": [
                {"label": "Total Revenue",   "value": "₹60,000",  "color": "var(--accent2)"},
                {"label": "Total Cost",      "value": "₹38,520",  "color": "var(--danger)"},
                {"label": "Gross Profit",    "value": "₹21,480",  "color": "var(--accent)"},
                {"label": "Per Unit Margin", "value": "₹21.48",   "color": "var(--warn)"}
            ],
            "chart": {
                "margin_pie": {"labels": ["Gross Margin", "Cost"], "values": [35.8, 64.2]},
                "cost_breakdown": {"labels": ["Raw Material", "Labour", "Delivery"], "values": [24000, 10000, 4520]}
            }
        }
    ]
}

# =====================================================
# HELPER — time series generator
# =====================================================

def generate_time_series(base_value, periods=6):
    labels, values = [], []
    now = datetime.now()
    for i in range(periods):
        dt = now - timedelta(days=30 * (periods - i))
        labels.append(dt.strftime("%b %Y"))
        noise = random.uniform(-0.05, 0.05)
        values.append(round(base_value * (1 + noise), 2))
    return labels, values

# =====================================================
# HELPER — send Gmail verification email
# =====================================================

def send_verification_email(dealer_email, dealer_name, token):
    verify_link = f"{BASE_URL}/verify-email?token={token}"

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = "Verify your SolarBag Dealer Email"
    msg["From"]    = GMAIL_USER
    msg["To"]      = dealer_email

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;background:#f4f6f8;padding:30px;">
      <div style="max-width:500px;margin:auto;background:white;border-radius:12px;padding:30px;
                  box-shadow:0 5px 15px rgba(0,0,0,0.1);">
        <h2 style="color:#f5a623;">SolarBag Dealer Verification</h2>
        <p>Hello <strong>{dealer_name}</strong>,</p>
        <p>You have been registered as a SolarBag dealer.
           Please verify your email by clicking the button below:</p>
        <a href="{verify_link}"
           style="display:inline-block;margin:20px 0;background:#f5a623;color:#000;
                  padding:12px 28px;border-radius:8px;font-weight:bold;text-decoration:none;">
          Verify My Email
        </a>
        <p style="color:#999;font-size:13px;">
          This link expires in 24 hours. If you did not request this, ignore this email.
        </p>
        <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
        <p style="color:#bbb;font-size:12px;">SolarBag | Jaipur, Rajasthan | support@solarbag.in</p>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, dealer_email, msg.as_string())
        print(f"Verification email sent to {dealer_email}")
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False

# =====================================================
# SEED — sample dealers (runs once if collection empty)
# =====================================================

def seed_dealers():
    if not mongodb_available:
        print("Skipping dealer seeding because MongoDB is unavailable.")
        return

    try:
        if dealers_collection.count_documents({}) == 0:
            sample_dealers = [
                {
                    "name": "SunPower Solar Solutions",
                    "business_type": "retailer",
                    "phone": "+91 98001 11001",
                    "email": "sunpower@example.com",
                    "city": "Jaipur", "state": "Rajasthan",
                    "address": "12, Tonk Road, Jaipur",
                    "email_verified": False,
                    "verify_token": secrets.token_urlsafe(32),
                    "created_at": datetime.utcnow()
                },
            {
                "name": "GreenWatt Distributors",
                "business_type": "distributor",
                "phone": "+91 98001 22002",
                "email": "greenwatt@example.com",
                "city": "Jodhpur", "state": "Rajasthan",
                "address": "45, Station Road, Jodhpur",
                "email_verified": False,
                "verify_token": secrets.token_urlsafe(32),
                "created_at": datetime.utcnow()
            },
            {
                "name": "SolarFit Installers",
                "business_type": "installer",
                "phone": "+91 98001 33003",
                "email": "solarfit@example.com",
                "city": "Udaipur", "state": "Rajasthan",
                "address": "8, Fateh Sagar Road, Udaipur",
                "email_verified": False,
                "verify_token": secrets.token_urlsafe(32),
                "created_at": datetime.utcnow()
            },
        ]
        dealers_collection.insert_many(sample_dealers)
        print("Sample dealers inserted into MongoDB.")
    except Exception as e:
        print(f"Dealer seeding failed: {e}")

seed_dealers()


# =====================================================
# USER AUTH / SESSION HELPERS
# =====================================================

auth_users = {
    "demo": "demo123"
}

def init_user_session(username, is_admin=False):
    session.clear()
    session.permanent = True
    session['username'] = username
    session['is_admin'] = bool(is_admin)
    session['solar_predictions'] = []
    session['paperbags_predictions'] = []
    session['predictions'] = []
    session.modified = True


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get('username'):
            return redirect(url_for('index'))
        return func(*args, **kwargs)
    return wrapper


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    if mongodb_available:
        user = users_collection.find_one({"username": username})
        if not user or user.get("password") != password:
            return jsonify({"error": "Invalid username or password."}), 401
        is_admin = bool(user.get("isAdmin", False))
    else:
        if auth_users.get(username) != password:
            return jsonify({"error": "Invalid username or password."}), 401
        is_admin = False

    init_user_session(username, is_admin=is_admin)
    return jsonify({"status": "success", "message": "Logged in successfully."})


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    if mongodb_available:
        if users_collection.find_one({"username": username}):
            return jsonify({"error": "Username already exists."}), 409
        users_collection.insert_one({
            "username": username,
            "password": password,
            "isAdmin": False,
            "created_at": datetime.utcnow()
        })
    else:
        if username in auth_users:
            return jsonify({"error": "Username already exists."}), 409
        auth_users[username] = password

    return jsonify({"status": "success", "message": "Account created. Please sign in."})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))


# =====================================================
# WEBSITE PAGES  (original routes — unchanged)
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
@login_required
def home():
    return render_template("Home.html")

@app.route("/prediction")
@login_required
def prediction():
    return render_template("prediction.html")

@app.route("/eco_analytics")
@login_required
def eco():
    return render_template("eco.html")

# =====================================================
# DASHBOARD PAGE
# =====================================================

@app.route("/dashboard")
@login_required
def dashboard():
    # Get session-specific predictions, fallback to global store
    solar = session.get('solar_predictions', [])
    paper = session.get('paperbags_predictions', [])
    is_admin = session.get('is_admin', False)
    
    # Only show current session data, not all historical data
    return render_template("dashboard.html",
        solar_latest  = solar[-1] if solar else None,
        solar_runs    = len(solar),
        solar_history = solar[-10:],
        paper_latest  = paper[-1] if paper else None,
        paper_runs    = len(paper),
        paper_history = paper[-10:],
        is_admin = is_admin
    )

# =====================================================
# CLEAR SESSION / NEW USER
# =====================================================

@app.route("/api/clear-session", methods=["POST"])
def clear_session():
    """Clear the current user's prediction history while keeping them logged in."""
    session['solar_predictions'] = []
    session['paperbags_predictions'] = []
    session['predictions'] = []
    session.modified = True
    return jsonify({"status": "success", "message": "Session cleared. Ready for new data."})

# =====================================================
# DASHBOARD JSON API
# =====================================================

@app.route("/api/dashboard")
def dashboard_api():
    # Get session-specific predictions instead of global store
    solar = session.get('solar_predictions', [])
    paper = session.get('paperbags_predictions', [])
    return jsonify({
        "solar":     {"latest": solar[-1] if solar else None, "total_runs": len(solar), "history": solar[-10:]},
        "paperbags": {"latest": paper[-1] if paper else None, "total_runs": len(paper), "history": paper[-10:]}
    })

# =====================================================
# SOLAR MARGIN PREDICTION API
# =====================================================

@app.route("/predict_solar", methods=["POST"])
def predict_solar():
    data = request.get_json()
    customer_name = data.get("customer_name", "").strip() or "Unnamed"

    inverter_cost  = float(data.get("inverter_cost", 0))
    install_cost   = float(data.get("install_cost", 0))
    selling_price  = float(data.get("selling_price", 0))
    no_of_plates   = float(data.get("no_of_plates", 0))
    plate_cost     = float(data.get("plate_cost", 0))

    total_plate_cost = no_of_plates * plate_cost
    total_cost       = inverter_cost + install_cost + total_plate_cost

    if selling_price <= 0:
        return jsonify({"error": "Selling price must be greater than 0"}), 400

    gross_margin_pct = ((selling_price - total_cost) / selling_price) * 100
    labels, values   = generate_time_series(gross_margin_pct)

    result = {
        "prediction": round(gross_margin_pct, 2),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "type":       "solar",
        "customer_name": customer_name,
        "inputs": {
            "inverter_cost":  inverter_cost,
            "install_cost":   install_cost,
            "selling_price":  selling_price,
            "no_of_plates":   no_of_plates,
            "plate_cost":     plate_cost
        },
        "breakdown": [
            {"label": "Total Revenue",    "value": f"Rs.{selling_price:,.0f}",              "color": "var(--accent2)"},
            {"label": "Total Cost",       "value": f"Rs.{total_cost:,.0f}",                 "color": "var(--danger)"},
            {"label": "Gross Profit",     "value": f"Rs.{selling_price - total_cost:,.0f}", "color": "var(--accent)"},
            {"label": "Plate Total Cost", "value": f"Rs.{total_plate_cost:,.0f}",           "color": "var(--muted)"}
        ],
        "chart": {
            "margin_pie": {
                "labels": ["Gross Margin", "Cost"],
                "values": [round(gross_margin_pct, 2), round(100 - gross_margin_pct, 2)]
            },
            "cost_breakdown": {
                "labels": ["Inverter", "Installation", "Plates"],
                "values": [inverter_cost, install_cost, total_plate_cost]
            },
            "trend": {"labels": labels, "values": values}
        }
    }

    # Store in session instead of global store
    if 'solar_predictions' not in session:
        session['solar_predictions'] = []
    session['solar_predictions'].append(result)
    session.modified = True
    
    # Also keep in global store for backward compatibility
    prediction_store["solar"].append(result)
    return jsonify(result)

# =====================================================
# SOLAR ENERGY ESTIMATION API
# =====================================================

@app.route("/predict_solar_energy", methods=["POST"])
def predict_solar_energy():
    data = request.get_json()

    irradiance  = float(data.get("irradiance", 800))
    temperature = float(data.get("temperature", 25))
    panel_area  = float(data.get("panel_area", 10))
    efficiency  = float(data.get("efficiency", 0.18))
    hours       = float(data.get("hours", 6))

    base_power     = irradiance * panel_area * efficiency
    daily_energy   = base_power * hours / 1000
    monthly_energy = daily_energy * 30
    temp_factor    = 1 - max(0, (temperature - 25) * 0.004)
    monthly_energy *= temp_factor
    prediction     = round(monthly_energy, 2)

    labels, values = generate_time_series(prediction)

    return jsonify({
        "prediction": prediction,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chart": {"labels": labels, "values": values}
    })

# =====================================================
# PAPER BAG GROSS MARGIN API
# =====================================================

@app.route("/predict_paperbags", methods=["POST"])
def predict_paperbags():
    data = request.get_json()
    customer_name = data.get("customer_name", "").strip() or "Unnamed"

    quantity      = float(data.get("quantity", 0))
    selling_price = float(data.get("selling_price", 0))
    raw_material  = float(data.get("raw_material", 0))
    labor_cost    = float(data.get("labor_cost", 0))
    delivery_cost = float(data.get("delivery_cost", 0))

    total_cost = raw_material + labor_cost + delivery_cost

    if selling_price <= 0:
        return jsonify({"error": "Selling price must be greater than 0"}), 400

    gross_margin_pct = ((selling_price - total_cost) / selling_price) * 100
    labels, values   = generate_time_series(gross_margin_pct)

    result = {
        "prediction": round(gross_margin_pct, 2),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "type":       "paperbags",
        "customer_name": customer_name,
        "inputs": {
            "quantity":      quantity,
            "selling_price": selling_price,
            "raw_material":  raw_material,
            "labor_cost":    labor_cost,
            "delivery_cost": delivery_cost
        },
        "breakdown": [
            {"label": "Total Revenue",   "value": f"Rs.{selling_price:,.0f}",                                                     "color": "var(--accent2)"},
            {"label": "Total Cost",      "value": f"Rs.{total_cost:,.0f}",                                                        "color": "var(--danger)"},
            {"label": "Gross Profit",    "value": f"Rs.{selling_price - total_cost:,.0f}",                                        "color": "var(--accent)"},
            {"label": "Per Unit Margin", "value": f"Rs.{(selling_price - total_cost) / quantity:.2f}" if quantity > 0 else "N/A", "color": "var(--warn)"}
        ],
        "chart": {
            "margin_pie": {
                "labels": ["Gross Margin", "Cost"],
                "values": [round(gross_margin_pct, 2), round(100 - gross_margin_pct, 2)]
            },
            "cost_breakdown": {
                "labels": ["Raw Material", "Labour", "Delivery"],
                "values": [raw_material, labor_cost, delivery_cost]
            },
            "trend": {"labels": labels, "values": values}
        }
    }

    # Store in session instead of global store
    if 'paperbags_predictions' not in session:
        session['paperbags_predictions'] = []
    session['paperbags_predictions'].append(result)
    session.modified = True
    
    # Also keep in global store for backward compatibility
    prediction_store["paperbags"].append(result)
    return jsonify(result)

# =====================================================
# STATS API
# =====================================================

@app.route("/api/stats")
def api_stats():
    # Get session-specific predictions
    solar = session.get('solar_predictions', [])
    paper = session.get('paperbags_predictions', [])

    avg_solar = round(sum(r["prediction"] for r in solar) / len(solar), 2) if solar else 0
    avg_paper = round(sum(r["prediction"] for r in paper) / len(paper), 2) if paper else 0

    return jsonify({
        "predictions_count":   len(solar) + len(paper),
        "solar_records_count": len(solar),
        "paper_records_count": len(paper),
        "avg_solar_margin":    avg_solar,
        "avg_paper_margin":    avg_paper,
        "last_updated":        datetime.now().strftime("%Y-%m-%d %H:%M")
    })

# =====================================================
# PREDICTIONS API
# =====================================================

@app.route("/api/predictions")
def api_predictions():
    limit = int(request.args.get("limit", 10))

    solar_tagged = [{**r, "business_type": "solar"}     for r in session.get('solar_predictions', [])]
    paper_tagged = [{**r, "business_type": "paperbags"} for r in session.get('paperbags_predictions', [])]

    all_preds = solar_tagged + paper_tagged
    all_preds.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return jsonify({"predictions": all_preds[:limit], "total": len(all_preds)})

@app.route("/api/support-message", methods=["POST"])
def submit_support_message():
    data = request.get_json() or {}
    name = data.get("name", "").strip() or "Anonymous"
    email = data.get("email", "").strip()
    message = data.get("message", "").strip()

    if not email or not message:
        return jsonify({"error": "Email and message are required."}), 400

    support_entry = {
        "name": name,
        "email": email,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "new"
    }

    support_messages.append(support_entry)
    try:
        support_collection.insert_one({**support_entry, "created_at": datetime.utcnow()})
    except Exception as e:
        print(f"Warning: could not save support message to MongoDB: {e}")

    return jsonify({"status": "success", "message": "Support request submitted."}), 201

@app.route("/api/support-messages", methods=["GET"])
@login_required
def get_support_messages():
    if not session.get('is_admin'):
        return jsonify({"error": "Forbidden"}), 403

    try:
        messages = list(support_collection.find({}, {"_id": 0}).sort("created_at", -1))
        return jsonify({"support_messages": messages})
    except Exception as e:
        print(f"Warning: could not load support messages from MongoDB: {e}")
        return jsonify({"support_messages": list(reversed(support_messages))})

# =====================================================
# NEW BUSINESS API — Multi-business support
# =====================================================

@app.route("/api/businesses")
def get_businesses():
    """Return all available business types and their configurations"""
    return jsonify(BUSINESS_TYPES)


def map_business_input_to_ml_features(business_type, form_data):
    if business_type == "solar":
        quantity = float(form_data.get("system_capacity", 1))
        unit_price = float(form_data.get("panel_cost", 0)) / max(quantity, 1)
        total_value = float(form_data.get("selling_price", 0))
        # Features must match training order: [Quantity, Unit_Price, Total_Value, type]
        return [quantity, unit_price, total_value, 1]

    if business_type == "paperbags":
        quantity = float(form_data.get("quantity", 1))
        unit_price = float(form_data.get("production_cost", 0))
        total_value = quantity * unit_price
        # Features must match training order: [Quantity, Unit_Price, Total_Value, type]
        return [quantity, unit_price, total_value, 0]

    raise ValueError("ML prediction is only available for solar and paperbags")


@app.route("/api/predict", methods=["POST"])
def predict_multi_business():
    """Generic prediction endpoint for any business type"""
    try:
        business_type = request.form.get("business_type", "").lower()
        
        if not business_type or business_type not in BUSINESS_TYPES:
            return jsonify({"error": "Invalid or missing business type"}), 400
        
        # Get business configuration
        config = BUSINESS_TYPES[business_type]
        fields = config["fields"]
        
        # Collect form data
        form_data = {}
        for field in fields:
            value = request.form.get(field["name"], "")
            if field["required"] and not value:
                return jsonify({"error": f"Missing required field: {field['label']}"}), 400
            
            # Convert to appropriate type
            if field["type"] == "number":
                try:
                    form_data[field["name"]] = float(value) if value else 0
                except:
                    return jsonify({"error": f"Invalid number for {field['label']}"}), 400
            else:
                form_data[field["name"]] = value
        
        model_info = get_model_info()
        accuracy = 0
        margin = None
        prediction_source = "Formula estimate"

        if business_type in {"solar", "paperbags"} and model_info.get("model_loaded"):
            try:
                features = map_business_input_to_ml_features(business_type, form_data)
                margin = predict_margin(features)
                accuracy = model_info.get("model_quality", 0) or 0
                prediction_source = "ML model"
            except Exception as ml_err:
                print(f"ML prediction failed, falling back to formula: {ml_err}")
                margin = calculate_margin(business_type, form_data)
        else:
            margin = calculate_margin(business_type, form_data)

        labels, values = generate_time_series(margin)

        result = {
            "margin": round(margin, 2),
            "business_type": business_type,
            "display_name": config["display_name"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "inputs": form_data,
            "accuracy": round(accuracy, 2) if accuracy else 0,
            "prediction_source": prediction_source,
            "trend": {"labels": labels, "values": values}
        }
        
        # Store in session
        if 'predictions' not in session:
            session['predictions'] = []
        session['predictions'].append(result)
        session.modified = True
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in predict_multi_business: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def calculate_margin(business_type, form_data):
    """Calculate margin based on business type with realistic variance"""
    import random

    variance = random.uniform(-4, 8)

    def clamp(value, minimum=0):
        return round(max(minimum, value), 2)

    if business_type == "electronics":
        quantity = form_data.get("quantity", 1)
        unit_price = form_data.get("unit_price", 0)
        selling_price = form_data.get("selling_price", 0)
        operational_cost = form_data.get("operational_cost", 0)

        total_cost = (unit_price * quantity) + operational_cost
        if selling_price <= 0:
            return clamp(10 + variance, 8)

        margin = ((selling_price - total_cost) / selling_price) * 100
        return clamp(margin + variance, 6)

    elif business_type == "solar":
        panel_cost = form_data.get("panel_cost", 0)
        inverter_cost = form_data.get("inverter_cost", 0)
        installation_cost = form_data.get("installation_cost", 0)
        selling_price = form_data.get("selling_price", 0)

        total_cost = panel_cost + inverter_cost + installation_cost
        if selling_price <= 0:
            return clamp(18 + variance, 10)

        margin = ((selling_price - total_cost) / selling_price) * 100
        return clamp(margin + variance, 8)

    elif business_type == "paperbags":
        quantity = form_data.get("quantity", 1)
        production_cost = form_data.get("production_cost", 0)
        selling_price = form_data.get("selling_price", 0)
        overhead_cost = form_data.get("overhead_cost", 0)

        per_unit_overhead = overhead_cost / max(quantity, 1)
        total_unit_cost = production_cost + per_unit_overhead
        if selling_price <= 0:
            return clamp(22 + variance, 15)

        margin = ((selling_price - total_unit_cost) / selling_price) * 100
        return clamp(margin + variance, 10)

    elif business_type == "apparel":
        quantity = form_data.get("quantity", 1)
        cost_per_unit = form_data.get("cost_per_unit", 0)
        selling_price = form_data.get("selling_price", 0)
        overhead = form_data.get("overhead", 0)

        total_revenue = selling_price * quantity
        total_cost = (cost_per_unit * quantity) + overhead
        if total_revenue <= 0:
            return clamp(16 + variance, 10)

        margin = ((total_revenue - total_cost) / total_revenue) * 100
        return clamp(margin + variance, 8)

    elif business_type == "food":
        daily_sales = form_data.get("daily_sales", 0)
        cogs_percentage = form_data.get("cogs_percentage", 30)
        fixed_costs = form_data.get("fixed_costs", 0)
        monthly_days = form_data.get("monthly_days", 30)

        daily_fixed = fixed_costs / max(monthly_days, 1)
        daily_cogs = daily_sales * (cogs_percentage / 100)
        total_daily_costs = daily_cogs + daily_fixed
        if daily_sales <= 0:
            return clamp(10 + variance, 5)

        margin = ((daily_sales - total_daily_costs) / daily_sales) * 100
        return clamp(margin + variance, 4)

    elif business_type == "retail":
        monthly_sales = form_data.get("monthly_sales", 0)
        avg_margin_percentage = form_data.get("avg_margin_percentage", 20)
        fixed_costs = form_data.get("fixed_costs", 0)
        variable_costs = form_data.get("variable_costs", 0)

        gross_profit = monthly_sales * (avg_margin_percentage / 100)
        net_profit = gross_profit - fixed_costs - variable_costs
        if monthly_sales <= 0:
            return clamp(12 + variance, 8)

        margin = (net_profit / monthly_sales) * 100
        return clamp(margin + variance, 5)

    elif business_type == "manufacturing":
        monthly_production = form_data.get("monthly_production", 1)
        raw_material_cost = form_data.get("raw_material_cost", 0)
        labor_cost = form_data.get("labor_cost", 0)
        selling_price = form_data.get("selling_price", 0)
        factory_overhead = form_data.get("factory_overhead", 0)

        total_revenue = selling_price * monthly_production
        total_cost = (raw_material_cost + labor_cost) * monthly_production + factory_overhead
        if total_revenue <= 0:
            return clamp(14 + variance, 10)

        margin = ((total_revenue - total_cost) / total_revenue) * 100
        return clamp(margin + variance, 8)

    elif business_type == "services":
        hourly_rate = form_data.get("hourly_rate", 0)
        monthly_hours = form_data.get("monthly_hours", 0)
        direct_costs = form_data.get("direct_costs", 0)
        overhead = form_data.get("overhead", 0)

        total_revenue = hourly_rate * monthly_hours
        total_costs = (direct_costs * monthly_hours) + overhead
        if total_revenue <= 0:
            return clamp(18 + variance, 10)

        margin = ((total_revenue - total_costs) / total_revenue) * 100
        return clamp(margin + variance, 8)

    else:
        return clamp(18 + variance, 10)

# =====================================================
# DEALER API — Add dealer + send verification email
# =====================================================

@app.route("/api/dealers/add", methods=["POST"])
def add_dealer():
    db_error = require_db()
    if db_error:
        return db_error

    data = request.get_json()

    required = ["name", "business_type", "phone", "email", "city", "address", "state"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing field: {field}"}), 400

    if dealers_collection.find_one({"email": data["email"]}):
        return jsonify({"error": "A dealer with this email already exists."}), 400

    token = secrets.token_urlsafe(32)

    dealer = {
        "name":           data["name"],
        "business_type":  data["business_type"].lower(),
        "phone":          data["phone"],
        "email":          data["email"],
        "city":           data["city"],
        "state":          data["state"],
        "address":        data["address"],
        "email_verified": False,
        "verify_token":   token,
        "created_at":     datetime.utcnow()
    }

    dealers_collection.insert_one(dealer)
    sent = send_verification_email(data["email"], data["name"], token)

    if sent:
        return jsonify({"message": f"Dealer added! Verification email sent to {data['email']}"}), 201
    else:
        return jsonify({"message": "Dealer added but email failed. Check GMAIL_USER and GMAIL_PASSWORD."}), 201

# =====================================================
# DEALER API — Email verification link
# =====================================================

@app.route("/verify-email")
def verify_email():
    db_error = require_db()
    if db_error:
        return db_error

    token = request.args.get("token", "")

    if not token:
        return "<h2>Invalid verification link.</h2>", 400

    dealer = dealers_collection.find_one({"verify_token": token})

    if not dealer:
        return "<h2>Link is invalid or already used.</h2>", 404

    if dealer.get("email_verified"):
        return """
        <html><body style='font-family:Arial;text-align:center;padding:60px'>
        <h2 style='color:#00c853'>Already verified!</h2>
        <p>Your dealer profile is active on SolarBag.</p>
        </body></html>
        """

    dealers_collection.update_one(
        {"verify_token": token},
        {"$set": {"email_verified": True, "verify_token": None}}
    )

    return """
    <html>
    <body style='font-family:Arial;text-align:center;padding:60px;background:#f4f6f8'>
      <div style='max-width:480px;margin:auto;background:white;padding:40px;border-radius:12px;
                  box-shadow:0 5px 15px rgba(0,0,0,0.1)'>
        <h2 style='color:#f5a623'>SolarBag</h2>
        <h3 style='color:#00c853'>Email Verified Successfully!</h3>
        <p>Your dealer profile is now <strong>active</strong> and visible to customers.</p>
        <a href='/' style='display:inline-block;margin-top:20px;background:#f5a623;color:#000;
                           padding:10px 24px;border-radius:8px;font-weight:bold;text-decoration:none'>
          Go to Dashboard
        </a>
      </div>
    </body>
    </html>
    """

# =====================================================
# DEALER API — Get verified dealers by business type
# =====================================================

@app.route("/api/dealers", methods=["GET"])
def get_dealers():
    db_error = require_db()
    if db_error:
        return db_error

    business_type = request.args.get("type", "").strip()

    if not business_type:
        return jsonify({"error": "Please provide a business type"}), 400

    query = {
        "business_type":  {"$regex": business_type, "$options": "i"},
        "email_verified": True   # only show verified dealers
    }

    dealers = list(dealers_collection.find(query, {"_id": 0, "verify_token": 0}))

    if not dealers:
        return jsonify({
            "dealers": [],
            "message": f"No verified dealers found for '{business_type}'"
        }), 200

    return jsonify({"dealers": dealers, "count": len(dealers)}), 200

# =====================================================
# DEALER API — Resend verification email
# =====================================================

@app.route("/api/dealers/resend-verification", methods=["POST"])
def resend_verification():
    db_error = require_db()
    if db_error:
        return db_error

    data  = request.get_json()
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    dealer = dealers_collection.find_one({"email": email})

    if not dealer:
        return jsonify({"error": "No dealer found with this email"}), 404

    if dealer.get("email_verified"):
        return jsonify({"message": "Email is already verified"}), 200

    token = secrets.token_urlsafe(32)
    dealers_collection.update_one({"email": email}, {"$set": {"verify_token": token}})

    sent = send_verification_email(email, dealer["name"], token)

    if sent:
        return jsonify({"message": "Verification email resent!"}), 200
    else:
        return jsonify({"error": "Failed to send. Check Gmail config in app.py."}), 500

# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/health")
def health():
    return jsonify({"status": "running"})

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

if __name__ == "__main__":
    app.run(debug=True, port=5000)