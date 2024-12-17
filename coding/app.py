from flask import Flask, request, render_template, send_file
import yfinance as yf
import numpy as np
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
import base64
import os
from fpdf import FPDF

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Path ke file inflasi (disesuaikan dengan struktur direktori)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INFLASI_PATH = os.path.join(BASE_DIR, "DataInflasi.xlsx")

# Fungsi untuk memprediksi menggunakan ARIMA (auto_arima)
def arima_predict(data, steps=1):
    model = auto_arima(data, seasonal=True, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=steps)
    return forecast

# Fungsi untuk membaca data inflasi dari file Excel
def process_inflation_data(file_path):
    df = pd.read_excel(file_path)
    if "Inflasi (%)" not in df.columns:
        raise ValueError("File harus memiliki kolom 'Inflasi (%)'")
    return df["Inflasi (%)"].dropna().values / 100  # Konversi ke desimal

@app.route("/")
def home():
    return render_template("index.html")

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
    img1.seek(0)
    plt.close()

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
    img2.seek(0)
    plt.close()

    return render_template(
        "result.html",
        plot_url1=f"data:image/png;base64,{plot_url1}",
        plot_url2=f"data:image/png;base64,{plot_url2}",
        stock_code=stock_code,
        table_data=table_data,
        total_investment_value=total_investment_value,
    )

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    stock_code = request.form["stock_code"]
    table_data = eval(request.form["table_data"])
    total_investment_value = request.form["total_investment_value"]

    img1 = io.BytesIO(base64.b64decode(request.form["plot_url1"].split(",")[1]))
    img2 = io.BytesIO(base64.b64decode(request.form["plot_url2"].split(",")[1]))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Hasil Prediksi Investasi Saham", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Kode Saham: {stock_code}", ln=True, align="C")

    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Rincian Perhitungan Bulanan", ln=True, align="L")
    pdf.cell(40, 10, txt="Bulan", border=1, align="C")
    pdf.cell(40, 10, txt="Tabungan (Rp)", border=1, align="C")
    pdf.cell(40, 10, txt="Harga Per Lot (Rp)", border=1, align="C")
    pdf.cell(40, 10, txt="Total Lot Dimiliki", border=1, align="C")
    pdf.cell(40, 10, txt="Nilai Investasi (Rp)", border=1, align="C")
    pdf.ln()
    for row in table_data:
        pdf.cell(40, 10, txt=str(row["month"]), border=1)
        pdf.cell(40, 10, txt=str(row["savings"]), border=1)
        pdf.cell(40, 10, txt=str(row["price_per_lot"]), border=1)
        pdf.cell(40, 10, txt=str(row["lots_owned"]), border=1)
        pdf.cell(40, 10, txt=str(row["investment_value"]), border=1)
        pdf.ln()

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Nilai Investasi pada Bulan Terakhir: Rp {total_investment_value}", ln=True)

    pdf.add_page()
    pdf.cell(200, 10, txt="Grafik Prediksi Nilai Investasi", ln=True, align="L")
    img_path1 = os.path.join(BASE_DIR, "temp_plot1.png")
    with open(img_path1, "wb") as f:
        f.write(img1.getvalue())
    pdf.image(img_path1, x=10, y=30, w=180)

    pdf.add_page()
    pdf.cell(200, 10, txt="Grafik Pengaruh Inflasi pada Tabungan dan Investasi", ln=True, align="L")
    img_path2 = os.path.join(BASE_DIR, "temp_plot2.png")
    with open(img_path2, "wb") as f:
        f.write(img2.getvalue())
    pdf.image(img_path2, x=10, y=30, w=180)

    pdf_output = os.path.join(BASE_DIR, "investment_report.pdf")
    pdf.output(pdf_output)

    os.remove(img_path1)
    os.remove(img_path2)

    return send_file(pdf_output, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
