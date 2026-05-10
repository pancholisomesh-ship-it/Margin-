# =====================================================
# BUSINESS CONFIGURATION
# =====================================================
# Defines all business types and their dynamic fields

BUSINESS_TYPES = {
    "electronics": {
        "display_name": "Electronics & Mobile Shops",
        "icon": "📱",
        "description": "Electronics and mobile device retail",
        "fields": [
            {
                "name": "product_category",
                "label": "Product Category",
                "type": "select",
                "options": ["Smartphones", "Laptops", "Tablets", "Accessories", "Mixed"],
                "required": True
            },
            {
                "name": "quantity",
                "label": "Quantity (Units)",
                "type": "number",
                "placeholder": "Enter quantity",
                "required": True
            },
            {
                "name": "unit_price",
                "label": "Unit Price (₹)",
                "type": "number",
                "placeholder": "Enter cost per unit",
                "required": True
            },
            {
                "name": "selling_price",
                "label": "Selling Price per Unit (₹)",
                "type": "number",
                "placeholder": "Enter selling price",
                "required": True
            },
            {
                "name": "operational_cost",
                "label": "Operational Cost (₹)",
                "type": "number",
                "placeholder": "Rent, electricity, staff, etc.",
                "required": True
            }
        ]
    },
    "solar": {
        "display_name": "Solar Energy Systems",
        "icon": "☀️",
        "description": "Solar panel installation and equipment",
        "fields": [
            {
                "name": "system_capacity",
                "label": "System Capacity (kW)",
                "type": "number",
                "placeholder": "e.g., 5, 10, 25",
                "required": True
            },
            {
                "name": "panel_cost",
                "label": "Panel Cost (₹)",
                "type": "number",
                "placeholder": "Total panel cost",
                "required": True
            },
            {
                "name": "inverter_cost",
                "label": "Inverter Cost (₹)",
                "type": "number",
                "placeholder": "Inverter price",
                "required": True
            },
            {
                "name": "installation_cost",
                "label": "Installation Cost (₹)",
                "type": "number",
                "placeholder": "Labor and mounting",
                "required": True
            },
            {
                "name": "selling_price",
                "label": "Total Project Value (₹)",
                "type": "number",
                "placeholder": "Final selling price",
                "required": True
            }
        ]
    },
    "paperbags": {
        "display_name": "Paper Bags Manufacturing",
        "icon": "📦",
        "description": "Paper bag production and sales",
        "fields": [
            {
                "name": "bag_type",
                "label": "Bag Type",
                "type": "select",
                "options": ["Standard", "Kraft", "Laminated", "Printed", "Custom"],
                "required": True
            },
            {
                "name": "quantity",
                "label": "Quantity (Bags)",
                "type": "number",
                "placeholder": "Number of bags",
                "required": True
            },
            {
                "name": "production_cost",
                "label": "Production Cost per Bag (₹)",
                "type": "number",
                "placeholder": "Material + labor cost",
                "required": True
            },
            {
                "name": "selling_price",
                "label": "Selling Price per Bag (₹)",
                "type": "number",
                "placeholder": "Market price",
                "required": True
            },
            {
                "name": "overhead_cost",
                "label": "Monthly Overhead (₹)",
                "type": "number",
                "placeholder": "Machinery, utilities, etc.",
                "required": True
            }
        ]
    },
    "apparel": {
        "display_name": "Apparel & Clothing",
        "icon": "👕",
        "description": "Clothing and fashion retail",
        "fields": [
            {
                "name": "clothing_type",
                "label": "Clothing Type",
                "type": "select",
                "options": ["Men", "Women", "Kids", "Accessories", "Mix"],
                "required": True
            },
            {
                "name": "quantity",
                "label": "Quantity (Units)",
                "type": "number",
                "placeholder": "Number of items",
                "required": True
            },
            {
                "name": "cost_per_unit",
                "label": "Cost per Unit (₹)",
                "type": "number",
                "placeholder": "Wholesale cost",
                "required": True
            },
            {
                "name": "selling_price",
                "label": "Selling Price per Unit (₹)",
                "type": "number",
                "placeholder": "Retail price",
                "required": True
            },
            {
                "name": "overhead",
                "label": "Store Overhead (₹)",
                "type": "number",
                "placeholder": "Rent, staff, utilities",
                "required": True
            }
        ]
    },
    "food": {
        "display_name": "Food & Beverage",
        "icon": "🍔",
        "description": "Restaurant, cafe, or food business",
        "fields": [
            {
                "name": "item_type",
                "label": "Item Type",
                "type": "select",
                "options": ["Fast Food", "Cafe", "Restaurant", "Bakery", "Delivery"],
                "required": True
            },
            {
                "name": "daily_sales",
                "label": "Daily Sales (₹)",
                "type": "number",
                "placeholder": "Average daily revenue",
                "required": True
            },
            {
                "name": "cogs_percentage",
                "label": "Cost of Goods (% of sales)",
                "type": "number",
                "placeholder": "e.g., 30 for 30%",
                "required": True
            },
            {
                "name": "fixed_costs",
                "label": "Monthly Fixed Costs (₹)",
                "type": "number",
                "placeholder": "Rent, utilities, staff",
                "required": True
            },
            {
                "name": "monthly_days",
                "label": "Operating Days/Month",
                "type": "number",
                "placeholder": "e.g., 25, 30",
                "required": True
            }
        ]
    },
    "retail": {
        "display_name": "General Retail",
        "icon": "🛒",
        "description": "General merchandise retail store",
        "fields": [
            {
                "name": "product_type",
                "label": "Product Type",
                "type": "select",
                "options": ["Grocery", "Hardware", "Beauty", "Toys", "Mixed"],
                "required": True
            },
            {
                "name": "monthly_sales",
                "label": "Monthly Sales (₹)",
                "type": "number",
                "placeholder": "Total monthly revenue",
                "required": True
            },
            {
                "name": "avg_margin_percentage",
                "label": "Average Margin (%)",
                "type": "number",
                "placeholder": "Industry standard margin",
                "required": True
            },
            {
                "name": "fixed_costs",
                "label": "Monthly Fixed Costs (₹)",
                "type": "number",
                "placeholder": "Rent, salaries, utilities",
                "required": True
            },
            {
                "name": "variable_costs",
                "label": "Monthly Variable Costs (₹)",
                "type": "number",
                "placeholder": "Marketing, transportation",
                "required": True
            }
        ]
    },
    "manufacturing": {
        "display_name": "Manufacturing",
        "icon": "🏭",
        "description": "Small to medium manufacturing business",
        "fields": [
            {
                "name": "product_name",
                "label": "Product Name",
                "type": "text",
                "placeholder": "What do you manufacture?",
                "required": True
            },
            {
                "name": "monthly_production",
                "label": "Monthly Production (Units)",
                "type": "number",
                "placeholder": "Units produced per month",
                "required": True
            },
            {
                "name": "raw_material_cost",
                "label": "Raw Material Cost per Unit (₹)",
                "type": "number",
                "placeholder": "Material cost only",
                "required": True
            },
            {
                "name": "labor_cost",
                "label": "Labor Cost per Unit (₹)",
                "type": "number",
                "placeholder": "Wage per unit",
                "required": True
            },
            {
                "name": "selling_price",
                "label": "Selling Price per Unit (₹)",
                "type": "number",
                "placeholder": "Market price",
                "required": True
            },
            {
                "name": "factory_overhead",
                "label": "Monthly Factory Overhead (₹)",
                "type": "number",
                "placeholder": "Utilities, maintenance, etc.",
                "required": True
            }
        ]
    },
    "services": {
        "display_name": "Services Business",
        "icon": "💼",
        "description": "Consulting, freelance, or service business",
        "fields": [
            {
                "name": "service_type",
                "label": "Service Type",
                "type": "select",
                "options": ["Consulting", "Freelance", "Repairs", "Cleaning", "Education"],
                "required": True
            },
            {
                "name": "hourly_rate",
                "label": "Hourly/Daily Rate (₹)",
                "type": "number",
                "placeholder": "Your service rate",
                "required": True
            },
            {
                "name": "monthly_hours",
                "label": "Monthly Billable Hours",
                "type": "number",
                "placeholder": "e.g., 160 for full-time",
                "required": True
            },
            {
                "name": "direct_costs",
                "label": "Direct Costs per Service (₹)",
                "type": "number",
                "placeholder": "Materials, travel, etc.",
                "required": True
            },
            {
                "name": "overhead",
                "label": "Monthly Overhead (₹)",
                "type": "number",
                "placeholder": "Office, software, insurance",
                "required": True
            }
        ]
    }
}

def get_business(business_id):
    """Get business configuration by ID"""
    return BUSINESS_TYPES.get(business_id.lower())

def get_all_businesses():
    """Get all available business types"""
    return list(BUSINESS_TYPES.keys())

def get_business_display_name(business_id):
    """Get friendly display name for business type"""
    config = get_business(business_id)
    return config["display_name"] if config else business_id

def get_business_fields(business_id):
    """Get input fields for a specific business"""
    config = get_business(business_id)
    return config["fields"] if config else []
