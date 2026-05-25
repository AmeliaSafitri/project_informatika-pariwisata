from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# MEMBUAT APLIKASI FLASK
# =====================================================

app = Flask(__name__)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("hotel_bookings_balanced.csv")

# =====================================================
# MEMILIH FITUR YANG DIGUNAKAN
# =====================================================

fitur = [
    'hotel',
    'lead_time',
    'arrival_date_month',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'reserved_room_type',
    'assigned_room_type',
    'booking_changes',
    'deposit_type',
    'customer_type',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests'
]

target = 'is_canceled'

df = df[fitur + [target]]

# =====================================================
# MEMBERSIHKAN DATA KOSONG
# =====================================================

df = df.dropna()

# =====================================================
# ENCODING DATA KATEGORIKAL
# =====================================================

encoder_dict = {}

kolom_kategori = [
    'hotel',
    'arrival_date_month',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type'
]

for kolom in kolom_kategori:

    le = LabelEncoder()

    df[kolom] = le.fit_transform(df[kolom])

    encoder_dict[kolom] = le

# =====================================================
# MEMISAHKAN FITUR DAN TARGET
# =====================================================

X = df.drop(columns=[target])

y = df[target]

# =====================================================
# NORMALISASI DATA
# =====================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# MEMBAGI DATA TRAINING DAN TESTING
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MEMBUAT MODEL RANDOM FOREST
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =====================================================
# TRAINING MODEL
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# EVALUASI MODEL
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

# =====================================================
# MEMBUAT GRAFIK CANCEL & BOOKING
# =====================================================

grafik_df = pd.read_csv("hotel_bookings_balanced.csv")

cancel = grafik_df[grafik_df['is_canceled'] == 1]

booking = grafik_df[grafik_df['is_canceled'] == 0]

cancel_bulan = cancel['arrival_date_month'].value_counts()

booking_bulan = booking['arrival_date_month'].value_counts()

plt.figure(figsize=(12,6))

plt.plot(cancel_bulan.index,
         cancel_bulan.values,
         marker='o')

plt.plot(booking_bulan.index,
         booking_bulan.values,
         marker='o')

plt.legend(['Cancel', 'Booking'])

plt.title('Grafik Booking dan Cancel per Bulan')

plt.xlabel('Bulan')

plt.ylabel('Jumlah')

plt.xticks(rotation=45)

if not os.path.exists('static'):
    os.makedirs('static')

plt.savefig('static/grafik.png')

plt.close()

# =====================================================
# HALAMAN HOME
# =====================================================

@app.route('/')
def home():

    return render_template(
        'index.html',
        accuracy=round(accuracy * 100, 2)
    )

# =====================================================
# PREDIKSI
# =====================================================

@app.route('/prediksi', methods=['POST'])

def prediksi():

    # ==========================================
    # MENGAMBIL INPUT DARI FORM
    # ==========================================

    hotel = request.form['hotel']

    lead_time = float(request.form['lead_time'])

    arrival_date_month = request.form['arrival_date_month']

    adults = int(request.form['adults'])

    children = int(request.form['children'])

    babies = int(request.form['babies'])

    meal = request.form['meal']

    country = request.form['country'].upper()

    market_segment = request.form['market_segment']

    deposit_type = request.form['deposit_type']

    customer_type = request.form['customer_type']

    adr = float(request.form['adr'])

    total_of_special_requests = int(
        request.form['total_of_special_requests']
    )

    # ==========================================
    # ENCODING INPUT
    # ==========================================

    hotel = encoder_dict['hotel'].transform([hotel])[0]

    arrival_date_month = encoder_dict[
        'arrival_date_month'
    ].transform([arrival_date_month])[0]

    meal = encoder_dict['meal'].transform([meal])[0]

    country = encoder_dict['country'].transform([country])[0]

    market_segment = encoder_dict[
        'market_segment'
    ].transform([market_segment])[0]

    deposit_type = encoder_dict[
        'deposit_type'
    ].transform([deposit_type])[0]

    customer_type = encoder_dict[
        'customer_type'
    ].transform([customer_type])[0]

    # ==========================================
    # MEMBUAT ARRAY DATA INPUT
    # ==========================================

    data = np.array([[

        hotel,
        lead_time,
        arrival_date_month,
        1,
        2,
        adults,
        children,
        babies,
        meal,
        country,
        market_segment,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        deposit_type,
        customer_type,
        adr,
        0,
        total_of_special_requests

    ]])

    # ==========================================
    # NORMALISASI INPUT
    # ==========================================

    data = scaler.transform(data)

    # ==========================================
    # PREDIKSI
    # ==========================================

    hasil = model.predict(data)[0]

    if hasil == 1:

        prediksi = "BOOKING BERPOTENSI CANCEL"

        keterangan = """
        Pelanggan memiliki kemungkinan tinggi
        melakukan pembatalan pemesanan hotel.
        """

    else:

        prediksi = "BOOKING BERPOTENSI CHECK-IN"

        keterangan = """
        Pelanggan memiliki kemungkinan
        datang/check-in ke hotel.
        """

    return render_template(
        'index.html',
        hasil=prediksi,
        keterangan=keterangan,
        accuracy=round(accuracy * 100, 2)
    )

# =====================================================
# MENJALANKAN FLASK
# =====================================================

if __name__ == '__main__':

    app.run(debug=True, port=5001)