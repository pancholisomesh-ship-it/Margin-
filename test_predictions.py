import urllib.request
import json

def test_prediction(business_type, params):
    """Test prediction API"""
    form_data = "&".join([f"business_type={business_type}"] + [f"{k}={v}" for k, v in params.items()])
    req = urllib.request.Request('http://localhost:5000/api/predict', data=form_data.encode())
    try:
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        return data['margin']
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test cases
tests = [
    ("Solar_1", "solar", {
        "system_capacity": "10",
        "panel_cost": "100000",
        "inverter_cost": "20000",
        "installation_cost": "10000",
        "selling_price": "150000"
    }),
    ("Solar_2", "solar", {
        "system_capacity": "50",
        "panel_cost": "250000",
        "inverter_cost": "50000",
        "installation_cost": "25000",
        "selling_price": "400000"
    }),
    ("Paperbags_1", "paperbags", {
        "bag_type": "Standard",
        "quantity": "1000",
        "production_cost": "5",
        "selling_price": "10",
        "overhead_cost": "5000"
    }),
    ("Paperbags_2", "paperbags", {
        "bag_type": "Kraft",
        "quantity": "5000",
        "production_cost": "3",
        "selling_price": "8",
        "overhead_cost": "8000"
    })
]

print("Testing predictions with retrained ML model:\n")
for name, btype, params in tests:
    margin = test_prediction(btype, params)
    print(f"{name}: {margin}%")
