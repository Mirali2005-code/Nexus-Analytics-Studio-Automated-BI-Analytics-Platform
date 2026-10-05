import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_ecommerce_data(num_records=15000, output_path="sample_data/ecommerce_analytics_data.csv"):
    np.random.seed(42)
    
    # Anchor date: realistic recent date (e.g. up to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Generate Customer IDs (some repeat, some one-time)
    unique_customers = [f"CUST-{i:05d}" for i in range(1, int(num_records * 0.45))]
    # Assign repeat propensity
    cust_weights = np.random.pareto(a=1.5, size=len(unique_customers))
    cust_weights /= cust_weights.sum()
    
    customer_ids = np.random.choice(unique_customers, size=num_records, p=cust_weights)
    
    # Pre-assign customer metadata (age, gender, country) for consistency per customer
    cust_meta = {}
    for cid in unique_customers:
        age = int(np.clip(np.random.normal(35, 12), 18, 75))
        gender = np.random.choice(["Female", "Male", "Non-Binary"], p=[0.52, 0.45, 0.03])
        region = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], p=[0.45, 0.30, 0.18, 0.07])
        cust_meta[cid] = (age, gender, region)
        
    ages = [cust_meta[cid][0] for cid in customer_ids]
    genders = [cust_meta[cid][1] for cid in customer_ids]
    regions = [cust_meta[cid][2] for cid in customer_ids]
    
    # Generate Dates
    days_delta = (end_date - start_date).days
    random_days = np.random.randint(0, days_delta, size=num_records)
    random_seconds = np.random.randint(0, 86400, size=num_records)
    order_dates = [start_date + timedelta(days=int(d), seconds=int(s)) for d, s in zip(random_days, random_seconds)]
    
    categories = ["Electronics", "Fashion & Apparel", "Home & Kitchen", "Beauty & Personal Care", "Fitness & Sports"]
    cat_weights = [0.28, 0.32, 0.18, 0.14, 0.08]
    selected_categories = np.random.choice(categories, size=num_records, p=cat_weights)
    
    # Pricing based on category
    cat_price_ranges = {
        "Electronics": (45.0, 850.0),
        "Fashion & Apparel": (18.0, 220.0),
        "Home & Kitchen": (25.0, 340.0),
        "Beauty & Personal Care": (12.0, 110.0),
        "Fitness & Sports": (20.0, 290.0)
    }
    
    unit_prices = []
    quantities = np.random.choice([1, 2, 3, 4, 5], size=num_records, p=[0.65, 0.20, 0.08, 0.04, 0.03])
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], size=num_records, p=[0.40, 0.20, 0.18, 0.12, 0.07, 0.03])
    
    for cat in selected_categories:
        low, high = cat_price_ranges[cat]
        price = round(np.random.uniform(low, high), 2)
        unit_prices.append(price)
        
    unit_prices = np.array(unit_prices)
    total_amounts = np.round(unit_prices * quantities * (1 - discounts), 2)
    
    payment_methods = np.random.choice(["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Buy Now Pay Later"], size=num_records, p=[0.48, 0.22, 0.15, 0.10, 0.05])
    order_status = np.random.choice(["Completed", "Delivered", "Cancelled", "Returned"], size=num_records, p=[0.60, 0.28, 0.07, 0.05])
    ratings = np.random.choice([1, 2, 3, 4, 5, np.nan], size=num_records, p=[0.04, 0.06, 0.15, 0.38, 0.32, 0.05]) # intentionally inject realistic nulls
    
    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(num_records)],
        "order_date": [d.strftime("%Y-%m-%d %H:%M:%S") for d in order_dates],
        "customer_id": customer_ids,
        "customer_age": ages,
        "customer_gender": genders,
        "region": regions,
        "category": selected_categories,
        "unit_price": unit_prices,
        "quantity": quantities,
        "discount_percent": (discounts * 100).astype(int),
        "total_amount": total_amounts,
        "payment_method": payment_methods,
        "order_status": order_status,
        "customer_rating": ratings
    })
    
    # Calculate lifetime order count per customer up to that order or overall
    cust_order_counts = df['customer_id'].value_counts().to_dict()
    df['customer_total_orders'] = df['customer_id'].map(cust_order_counts)
    df['is_repeat_customer'] = df['customer_total_orders'].apply(lambda x: "Yes" if x > 1 else "No")
    
    # Introduce a few realistic dirty data quirks to demonstrate auto-cleaning
    # 1. A few duplicate rows
    duplicates = df.iloc[:35].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. A few missing values in payment_method
    null_indices = np.random.choice(df.index, size=45, replace=False)
    df.loc[null_indices, "payment_method"] = np.nan
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sample e-commerce records at {output_path}")
    return df

if __name__ == "__main__":
    create_sample_ecommerce_data()
