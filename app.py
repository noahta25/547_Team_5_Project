from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
import boto3
from dotenv import load_dotenv
import numpy as np
from prophet import Prophet

load_dotenv()

# --------------------------
# Step 1: Load trained Prophet model
# --------------------------
model_path = "sales_forecast_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load Prophet model: {e}")

# --------------------------
# Step 2: Load raw sales CSV from DigitalOcean Spaces
# --------------------------
SPACES_KEY = os.getenv("SPACES_KEY")
SPACES_SECRET = os.getenv("SPACES_SECRET")
SPACES_REGION = os.getenv("SPACES_REGION")
SPACES_BUCKET = os.getenv("SPACES_BUCKET")
SPACES_FILE = os.getenv("SPACES_FILE")

try:
    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=f"https://{SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=SPACES_KEY,
        aws_secret_access_key=SPACES_SECRET,
    )
    obj = client.get_object(Bucket=SPACES_BUCKET, Key=SPACES_FILE)
    sales_data = pd.read_csv(obj["Body"])
except Exception as e:
    raise RuntimeError(f"Failed to load CSV from Spaces: {e}")

# --------------------------
# Detect date and sales columns
# --------------------------
def detect_date_column(df):
    date_candidates = [col for col in df.columns if "date" in col.lower() or "day" in col.lower()]
    for col in date_candidates:
        try:
            pd.to_datetime(df[col])
            return col
        except Exception:
            continue
    # fallback to first datetime convertible column
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except Exception:
            continue
    raise ValueError("No date column found")

def detect_sales_column(df):
    sales_candidates = [col for col in df.columns if "sales" in col.lower() or "sale" in col.lower() or "qty" in col.lower() or "quantity" in col.lower()]
    for col in sales_candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    # fallback to first numeric column that's not date
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_datetime(df[col])
            except Exception:
                return col
    raise ValueError("No sales column found")

DATE_COL = detect_date_column(sales_data)
SALES_COL = detect_sales_column(sales_data)

print(f"Detected date column: {DATE_COL}")
print(f"Detected sales column: {SALES_COL}")

# --------------------------
# Step 3: Preprocess data for Prophet
# --------------------------

# Keep only necessary columns
df = sales_data[[DATE_COL, SALES_COL, 'article']].copy()

# Convert to datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# Drop rows with invalid dates
df = df.dropna(subset=[DATE_COL])

# Sort by date
df = df.sort_values(DATE_COL)

# Aggregate to daily totals per article
daily = (
    df.groupby([pd.Grouper(key=DATE_COL, freq="D"), 'article'])[SALES_COL]
      .sum()
      .rename("y")
      .reset_index()
      .rename(columns={DATE_COL: "ds"})
)

# Fill missing dates with zero sales for each article
all_days = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
articles = daily['article'].unique()
full_index = pd.MultiIndex.from_product([all_days, articles], names=['ds', 'article'])
daily = (
    daily.set_index(['ds', 'article'])
         .reindex(full_index, fill_value=0.0)
         .reset_index()
)

# Ensure no negative sales
daily["y"] = daily["y"].clip(lower=0)

# Assign season to each date
def assign_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

daily["season"] = daily["ds"].dt.month.apply(assign_season)

# Replace sales_data with daily preprocessed dataframe
sales_data = daily

print(f"Date range: {sales_data['ds'].min().date()} → {sales_data['ds'].max().date()}")
print(f"Total days: {len(sales_data)}")

# --------------------------
# Step 4: Initialize Flask app
# --------------------------
app = Flask(__name__)

def format_table(header, rows, col_widths=None):
    # header: list of column names
    # rows: list of dicts with keys matching header
    # col_widths: optional list of int for column widths, else auto
    if not rows:
        # No data, just show header
        if col_widths is None:
            col_widths = [len(h) for h in header]
        header_line = "  ".join(h.ljust(w) for h, w in zip(header, col_widths))
        return header_line + "\n(No data)"
    if col_widths is None:
        col_widths = []
        for h in header:
            max_len = len(h)
            for row in rows:
                val = row.get(h, "")
                val_str = f"{val}" if val is not None else ""
                if len(val_str) > max_len:
                    max_len = len(val_str)
            col_widths.append(max_len)
    header_line = "  ".join(h.ljust(w) for h, w in zip(header, col_widths))
    lines = [header_line]
    for row in rows:
        line = "  ".join(f"{str(row.get(h,''))}".ljust(w) for h, w in zip(header, col_widths))
        lines.append(line)
    return "\n".join(lines)

@app.route("/")
def home():
    seasons = sorted(sales_data["season"].unique().tolist())
    view_options = ["Top 10", "Bottom 10", "All"]
    yes_no = ["Yes", "No"]
    return render_template(
        "index.html", seasons=seasons, view_options=view_options, yes_no=yes_no
    )

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.json
    season = data.get("season")
    view = data.get("view")
    historical = data.get("historical") == "Yes"
    predicted = data.get("predicted") == "Yes"

    output_sections = []

    # Filter by season
    season_data = sales_data[sales_data["season"] == season]

    # Historical data: total sales aggregated per article and filtered by view option
    if historical:
        # Aggregate total sales per article
        article_totals = season_data.groupby("article")["y"].sum().reset_index()

        if view == "Top 10":
            article_totals = article_totals.nlargest(10, "y")
        elif view == "Bottom 10":
            article_totals = article_totals.nsmallest(10, "y")
        # else "All" no change

        # Format as table string
        hist_rows = []
        for _, row in article_totals.iterrows():
            hist_rows.append({"Article": row["article"], "Total Sold": int(round(row["y"]))})
        hist_table = format_table(["Article", "Total Sold"], hist_rows)
        output_sections.append("Historical Sales:\n" + hist_table)

    # Predicted data using trained Prophet model: total predicted sales per season
    if predicted:
        # Generate future dates using the loaded Prophet model
        future = model.make_future_dataframe(periods=90, freq="D")
        forecast_df = model.predict(future)
        forecast_df["season"] = forecast_df["ds"].dt.month.apply(assign_season)
        
        # Filter by requested season
        forecast_season = forecast_df[forecast_df["season"] == season]
        
        # Total predicted sales
        predicted_total_sales = forecast_season["yhat"].sum()

        if view in ["Top 10", "Bottom 10"]:
            # Use historical totals to calculate proportions
            article_totals = season_data.groupby("article")["y"].sum().reset_index()
            if view == "Top 10":
                article_totals = article_totals.nlargest(10, "y")
            else:  # Bottom 10
                article_totals = article_totals.nsmallest(10, "y")

            total_historical_sales = article_totals["y"].sum()
            if total_historical_sales > 0:
                article_totals["predicted_sales"] = article_totals["y"] / total_historical_sales * predicted_total_sales
            else:
                article_totals["predicted_sales"] = 0.0

            pred_rows = []
            for _, row in article_totals.iterrows():
                pred_rows.append({"Article": row["article"], "Predicted Sold": int(round(row["predicted_sales"]))})
            pred_table = format_table(["Article", "Predicted Sold"], pred_rows)
            output_sections.append("Predicted Sales:\n" + pred_table)
        else:  # "All"
            pred_rows = [{"Total Predicted Sales": int(round(predicted_total_sales))}]
            pred_table = format_table(["Total Predicted Sales"], pred_rows)
            output_sections.append("Predicted Sales:\n" + pred_table)

    return "\n\n".join(output_sections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)