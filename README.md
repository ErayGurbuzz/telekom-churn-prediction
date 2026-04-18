# 📊 Telco Customer Churn: End-to-End Prediction & API Service

## 🚀 Proje Genel Bakış

Kaggle üzerindeki **Telco Customer Churn** veri seti kullanılarak müşteri kaybı tahmini yapan bir makine öğrenmesi modeli ve bu modeli servis eden basit bir API uygulaması içerir.
Sistem, ham verinin işlenmesinden başlayarak, 5 farklı gelişmiş algoritmanın yarıştırılması, modelin bir API (FastAPI) arkasına gizlenmesi ve son kullanıcı için bir arayüz (Streamlit) sunulmasına kadar olan tüm aşamaları kapsar.

## Proje Kapsamı

- Veri temizleme ve ön işleme
- En az iki model ile karşılaştırma
- En iyi modeli seçme
- Seçilen modeli API servisi haline getirme

**Temel Yetkinlikler:**

- **Gelişmiş Model Yarıştırma:** Logistic Regression, Random Forest, XGBoost, LightGBM ve CatBoost modelleri arasından en iyi F1-skoruna sahip olanı otomatik seçer.
- **Dinamik Sınıf Dengeleme:** Verideki azınlık sınıfı (Churn) için `scale_pos_weight` ve `class_weight` parametrelerini otomatik hesaplar.
- **Model Yarıştırma:** XGBoost, LightGBM, CatBoost ve Random Forest modellerinin benchmark analizi.
- **Modern Mimari:** **FastAPI** ile asenkron Backend ve **Streamlit** ile interaktif Frontend deneyimi sunar.
- **Robust Preprocessing:** Scikit-learn Pipeline yapısı ile veri sızıntısı (leakage) engellenmiş, ölçeklenebilir bir yapı kurulmuştur.

## 📂 Klasör Yapısı

```bash
churn_challenge/
├── data/               # Ham veri seti
├── models/             # Şampiyon model (pkl) ve metrik raporları (json)
├── src/
│   ├── train.py        # Model Tournament & Training Engine
│   ├── app.py          # FastAPI Backend Service
│   ├── predict.py      # Inference Logic & Model Loader
│   └── frontend.py     # Streamlit Web UI
├── requirements.txt    # Bağımlılıklar
└── README.md           # Proje Rehberi
└── REPORT.md           # Teknik Analiz Raporu
```

## Veri Seti

Kullanılan veri seti: **Telco Customer Churn**
Dosya adı:

```bash
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Bu dosyayı `data/` klasörü altına koymanız gerekir.

## Kurulum

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt #Kütüphane yüklemek
```

## Model Eğitimi

```bash
python src/train.py # Modelleri yarıştır ve en iyisini kaydet
```

Bu komut çalıştıktan sonra:

- `model/churn_pipeline.pkl`
- `model/metrics.json`

oluşacaktır.

## API'yi Çalıştırma

```bash
uvicorn src.app:app --reload
```

http://127.0.0.1:8000/docs üzerinden tüm parametreleri test edebilirsiniz.
API açıldıktan sonra:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

# Kullanıcı arayüzünü aç

```bash
streamlit run src/frontend.py
```

## Predict Endpoint

### Endpoint

```http
POST /predict
```

### Arayüzsüz Örnek Request Body

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1050.2
}
```

### Örnek Response

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability": 0.8123
}
```

## Modelleme Yaklaşımı

Bu çalışma kapsamında beş farklı model denenmiştir:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

Model seçimi için şu metrikler takip edilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Geliştirme Notları

- `TotalCharges` numerik tipe çevrilmiştir.
- `customerID` model girdisinden çıkarılmıştır.
- Kategorik değişkenler için `OneHotEncoder` kullanılmıştır.
- Sayısal değişkenler için `StandardScaler` uygulanmıştır.
- `train_test_split(..., stratify=y)` ile veri bölünmüştür.

## Geliştirme Önerileri

- Hyperparameter tuning eklenebilir.
- CI/CD ve test altyapısı eklenebilir.
