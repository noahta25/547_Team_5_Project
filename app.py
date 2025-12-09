from flask import Flask, request, render_template
import pickle
import os
import boto3
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# --------------------------
# Step 1: Load trained Prophet model from DigitalOcean Spaces
# --------------------------
SPACES_KEY = os.getenv("SPACES_KEY")
SPACES_SECRET = os.getenv("SPACES_SECRET")
SPACES_REGION = os.getenv("SPACES_REGION")
SPACES_BUCKET = os.getenv("SPACES_BUCKET")

try:
    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=f"https://{SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=SPACES_KEY,
        aws_secret_access_key=SPACES_SECRET,
    )
    # List all .pkl files in the bucket
    response = client.list_objects_v2(Bucket=SPACES_BUCKET)
    pkl_files = [obj for obj in response.get('Contents', []) if obj['Key'].endswith('.pkl')]
    if not pkl_files:
        raise RuntimeError("No .pkl files found in the bucket")
    # Select the latest based on LastModified
    latest_pkl = max(pkl_files, key=lambda x: x['LastModified'])
    print(f"Loading Prophet model from file: {latest_pkl['Key']}")
    obj = client.get_object(Bucket=SPACES_BUCKET, Key=latest_pkl['Key'])
    model_artifact = pickle.load(obj['Body'])
    trained_models = model_artifact.get("models", {})
    df_items = model_artifact.get("historical_data", pd.DataFrame())
    ITEM_COL = model_artifact.get("item_col", None)
except Exception as e:
    raise RuntimeError(f"Failed to load Prophet model from Spaces: {e}")

# --------------------------
# Assign season to each date
# --------------------------
def assign_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

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
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    view_options = ["All", "Top 10", "Bottom 10"]
    yes_no = ["No", "Yes"]
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

    # Historical data processing
    if historical:
        if df_items.empty:
            output_sections.append("Historical Sales Data:\n(No historical data available)")
        else:
            df_hist = df_items.copy()
            df_hist['season'] = df_hist['ds'].dt.month.apply(assign_season)
            if season != "All":
                df_hist = df_hist[df_hist['season'] == season]
            if ITEM_COL and ITEM_COL in df_hist.columns:
                agg_hist = df_hist.groupby(ITEM_COL)['y'].sum().reset_index()
                agg_hist.rename(columns={ITEM_COL: 'Product', 'y': 'Sales'}, inplace=True)
            else:
                total_sales = df_hist['y'].sum()
                agg_hist = pd.DataFrame([{'Product': 'All Products', 'Sales': total_sales}])

            # Apply view filter
            if view == "Top 10":
                agg_hist = agg_hist.sort_values(by='Sales', ascending=False).head(10)
            elif view == "Bottom 10":
                agg_hist = agg_hist.sort_values(by='Sales', ascending=True).head(10)
            else:  # All
                agg_hist = agg_hist.sort_values(by='Sales', ascending=False)

            hist_rows = agg_hist.to_dict(orient='records')
            hist_table = format_table(['Product', 'Sales'], hist_rows)
            output_sections.append("Historical Sales Data:\n" + hist_table)

    # Predicted data processing
    if predicted:
        if not trained_models:
            output_sections.append("Predicted Sales:\n(No predicted data available)")
        else:
            forecast_rows = []
            for product, prod_model in trained_models.items():
                # Generate future dataframe for 90 days
                future = prod_model.make_future_dataframe(periods=90, freq="D")
                future = future.copy()
                future['season'] = future['ds'].dt.month.apply(assign_season)

                # Filter future dataframe for selected season if not "All"
                if season != "All":
                    future_season = future[future['season'] == season]
                else:
                    future_season = future

                if future_season.empty:
                    total_pred = 0
                else:
                    forecast_df = prod_model.predict(future_season)
                    total_pred = forecast_df['yhat'].sum()

                forecast_rows.append({'Product': product, 'Predicted Sales': int(round(total_pred))})

            pred_df = pd.DataFrame(forecast_rows)

            # Apply view filter
            if view == "Top 10":
                pred_df = pred_df.sort_values(by='Predicted Sales', ascending=False).head(10)
            elif view == "Bottom 10":
                pred_df = pred_df.sort_values(by='Predicted Sales', ascending=True).head(10)
            else:  # All
                pred_df = pred_df.sort_values(by='Predicted Sales', ascending=False)

            pred_rows = pred_df.to_dict(orient='records')
            pred_table = format_table(['Product', 'Predicted Sales'], pred_rows)
            output_sections.append("Predicted Sales:\n" + pred_table)

    return "\n\n".join(output_sections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)