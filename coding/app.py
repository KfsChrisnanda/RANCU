from flask import Flask, request, render_template, send_file, make_response
import yfinance as yf
import numpy as np
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
import base64
import os
from matplotlib.backends.backend_pdf import PdfPages

# Initialize Flask app
app = Flask(__name__)

# Path to inflation data file (adjust based on your directory structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INFLASI_PATH = os.path.join(BASE_DIR, "DataInflasi.xlsx")

# List of stock codes
STOCK_CODES = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "UNTR.JK", "ARTO.JK", "BBHI.JK", "BBYB.JK", "KLBF.JK", 
    "SILO.JK", "TSPC.JK", "SRAJ.JK", "SAME.JK", "PYFA.JK", "INTP.JK", "INKP.JK", "SMGR.JK", "BRPT.JK", 
    "BRMS.JK", "MDKA.JK", "INTD.JK", "DPNS.JK", "EKAD.JK", "IGAR.JK", "AKPI.JK", "SRSN.JK", "ADMG.JK", 
    "SULI.JK", "TPIA.JK", "BAJA.JK", "SMDR.JK", "BIRD.JK", "AKSI.JK", "EMTK.JK", "MLPT.JK", "DLTA.JK", 
    "JPFA.JK", "ULTJ.JK", "MPPA.JK", "INDF.JK", "AISA.JK", "AALI.JK", "TBLA.JK", "SGRO.JK", "AMRT.JK", 
    "MIDI.JK", "WIIM.JK", "DSNG.JK", "SSMS.JK", "HERO.JK", "SDPC.JK", "SMAR.JK", "SKLT.JK", "TCID.JK", 
    "ADES.JK", "EPMT.JK", "PSDN.JK", "CEKA.JK", "STTP.JK", "WAPO.JK", "CPRO.JK", "ROTI.JK", "RANC.JK", 
    "SKBM.JK", "KBLI.JK", "MLIA.JK", "IMPC.JK", "VOKS.JK", "IKBI.JK", "ZBRA.JK", "KBLM.JK", "TIRA.JK", 
    "INDX.JK", "DYAN.JK", "APII.JK", "BUMI.JK", "PTRO.JK", "DOID.JK", "RAJA.JK", "ELSA.JK", "DSSA.JK", 
    "HRUM.JK", "WINS.JK", "ABMM.JK", "TPMA.JK", "SOCI.JK", "ITMA.JK", "RUIS.JK", "GTBO.JK", "BIPI.JK", 
    "BULL.JK", "PTIS.JK", "BSSR.JK", "BBRM.JK", "LEAD.JK", "GJTL.JK", "BMTR.JK", "PANR.JK", "GWSA.JK", 
    "ARGO.JK", "BAYU.JK", "BIMA.JK", "BOLT.JK", "LPIN.JK", "INDS.JK", "ERTX.JK", "BRAM.JK", "SONA.JK", 
    "FAST.JK", "LMPI.JK", "JSPT.JK", "TMPO.JK", "FORU.JK", "UNIT.JK", "GEMA.JK", "PJAA.JK", "BUVA.JK", 
    "ECII.JK", "CINT.JK", "SSIA.JK", "TOTL.JK", "PTPP.JK", "TBIG.JK", "CASS.JK", "BALI.JK", "DGIK.JK", 
    "NRCA.JK", "BUKK.JK", "DILD.JK", "PUDP.JK", "APLN.JK", "LPLI.JK", "JRPT.JK", "FMII.JK", "GMTD.JK", 
    "INPP.JK", "GPRA.JK", "BAPA.JK"
]

# Function to predict using ARIMA (auto_arima)
def arima_predict(data, steps=1):
    model = auto_arima(data, seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=steps)
    return forecast

# Function to process inflation data from Excel file
def process_inflation_data(file_path):
    df = pd.read_excel(file_path)
    if "Inflasi (%)" not in df.columns:
        raise ValueError("File harus memiliki kolom 'Inflasi (%)'")
    return df["Inflasi (%)"].dropna().values / 100  # Convert to decimal


@app.route("/")
def start():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")  

@app.route("/faq")
def faq():
    return render_template("faq.html")  

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    income = int(request.form["income"])
    saving_percent = float(request.form["saving_percent"]) / 100
    stock_code = request.form["stock_code"]
    saving_months = int(request.form["saving_months"])
    predict_months = int(request.form["predict_months"])

    try:
        stock_data = yf.download(stock_code, period="10y", interval="1mo")
        stock_prices = stock_data["Close"].dropna()
        if len(stock_prices) < 12:
            return "Data saham terlalu sedikit untuk prediksi."
    except Exception as e:
        return f"Error saat mengunduh data saham: {str(e)}"

    try:
        inflation_data = process_inflation_data(DATA_INFLASI_PATH)
    except Exception as e:
        return f"Error saat memproses data inflasi: {str(e)}"

    latest_prices = stock_prices.values
    predicted_prices = arima_predict(latest_prices, steps=predict_months)
    predicted_inflation = arima_predict(inflation_data, steps=predict_months)

    current_savings = 0
    total_lots = 0
    total_tabungan = 0
    table_data = []
    adjusted_savings = []
    adjusted_tabungan_only = []

    for month in range(predict_months):
        if month < saving_months:
            tabungan_bulanan = income * saving_percent
            current_savings += tabungan_bulanan
            total_tabungan += tabungan_bulanan

        price_per_lot = predicted_prices[month] * 100
        lots_bought = 0
        while current_savings >= price_per_lot:
            lots_bought += 1
            current_savings -= price_per_lot

        total_lots += lots_bought
        investment_value = total_lots * predicted_prices[month] * 100

        if month > 0:
            current_savings *= (1 - predicted_inflation[month - 1])

        table_data.append({
            "month": month + 1,
            "savings": round(current_savings, 2),
            "price_per_lot": round(price_per_lot, 2),
            "lots_owned": total_lots,
            "investment_value": round(investment_value, 2),
        })

        adjusted_savings.append(current_savings + investment_value)
        adjusted_tabungan_only.append(total_tabungan)

    total_investment_value = table_data[-1]["investment_value"]

    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, predict_months + 1),
        [row["investment_value"] for row in table_data],
        label="Nilai Investasi",
        color="orange",
    )
    plt.xlabel("Bulan")
    plt.ylabel("Nilai Investasi (Rp)")
    plt.title(f"Prediksi Nilai Investasi ({stock_code})")
    plt.legend()
    plt.grid()
    img1 = io.BytesIO()
    plt.savefig(img1, format="png")
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()

    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, predict_months + 1),
        adjusted_savings,
        label="Total Tabungan + Investasi (Dikenai Inflasi)",
        color="blue",
    )
    plt.plot(
        range(1, predict_months + 1),
        adjusted_tabungan_only,
        label="Hanya Tabungan (Tanpa Pengurangan Biaya Saham)",
        color="green",
    )
    plt.xlabel("Bulan")
    plt.ylabel("Nilai (Rp)")
    plt.title(f"Pengaruh Inflasi pada Tabungan dan Investasi ({stock_code})")
    plt.legend()
    plt.grid()
    img2 = io.BytesIO()
    plt.savefig(img2, format="png")
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()

    df_table_data = pd.DataFrame(table_data)
    
    # Convert the DataFrame to CSV and pass it as a query parameter to the result page
    table_data_csv = df_table_data.to_csv(index=False)  # Convert table to CSV string
    
    return render_template(
        "result.html",
        plot_url1=f"data:image/png;base64,{plot_url1}",
        plot_url2=f"data:image/png;base64,{plot_url2}",
        stock_code=stock_code,
        table_data=table_data,
        total_investment_value=total_investment_value,
        table_data_csv=table_data_csv,  # Pass CSV data to the result page
    )

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/download_csv")
def download_csv():
    csv_data = request.args.get("csv_data")
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)
