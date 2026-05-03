# 📊 Telco Customer Churn: End-to-End Prediction & API Service

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

## 🚀 Proje Genel Bakış
Kaggle üzerindeki **Telco Customer Churn** veri seti kullanılarak müşteri kaybı tahmini yapan bir makine öğrenmesi modeli ve bu modeli servis eden tam kapsamlı bir API uygulamasıdır. Sistem, ham verinin işlenmesinden başlayarak, 5 farklı gelişmiş algoritmanın yarıştırılması, şampiyon modelin bir API (FastAPI) arkasına alınması ve son kullanıcı için interaktif bir arayüz (Streamlit) sunulmasına kadar olan tüm aşamaları kapsar.

## ✨ Proje Kapsamı ve Temel Yetkinlikler
* **Kapsamlı Veri Ön İşleme:** Scikit-learn Pipeline yapısı ile veri sızıntısı (leakage) engellenmiş, tamamen ölçeklenebilir bir yapı kurulmuştur.
* **Gelişmiş Model Yarıştırma:** Logistic Regression, Random Forest, XGBoost, LightGBM ve CatBoost modelleri eğitilmiş, en iyi F1-skoruna sahip olan otomatik seçilmiştir.
* **Dinamik Sınıf Dengeleme:** Verideki azınlık sınıfı (Churn) için `scale_pos_weight` ve `class_weight` parametreleri otomatik hesaplanmıştır.
* **Modern Mimari:** **FastAPI** ile asenkron Backend ve **Streamlit** ile interaktif Frontend deneyimi sunulmaktadır.

## 📈 Model Performansı
Bu çalışma kapsamında beş farklı model denenmiş ve aşağıdaki metrikler (Accuracy, Precision, Recall, F1-score, ROC-AUC) takip edilmiştir. Model yarıştırması sonucunda en yüksek performansı **Random Forest** göstermiştir:

| Model | Doğruluk (Accuracy) | F1-Skoru | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Random Forest** | %75.5 | %63.3 | 0.835 |
| CatBoost | %75.6 | %62.0 | 0.837 |
| XGBoost | %74.5 | %61.4 | 0.834 |
| Logistic Regression | %73.8 | %61.3 | 0.841 |

## 📂 Klasör Yapısı

```text
churn_challenge/
├── data/               # Ham veri seti (Kaggle)
├── models/             # Şampiyon model (pkl) ve metrik raporları (json)
├── src/
│   ├── train.py        # Model Tournament & Training Engine
│   ├── app.py          # FastAPI Backend Service
│   ├── predict.py      # Inference Logic & Model Loader
│   └── frontend.py     # Streamlit Web UI
├── requirements.txt    # Bağımlılıklar
├── README.md           # Proje Rehberi
└── REPORT.md           # Teknik Analiz Raporu
```


## 💾 Veri Seti
Kullanılan veri seti: Telco Customer Churn

Dosya adı: WA_Fn-UseC_-Telco-Customer-Churn.csv

Kurulum sonrası bu dosyayı data/ klasörü altına yerleştirmeniz gerekmektedir.

## 🛠️ Kurulum ve Çalıştırma
### 1. Sanal Ortam Hazırlığı:

```bash
  python -m venv venv
  .\venv\Scripts\activate  # Windows için
  # source venv/bin/activate # macOS/Linux için
  pip install -r requirements.txt
```

### 2. Model Eğitimi (Opsiyonel)
Aşağıdaki komut modelleri yarıştırır, en iyisini seçer ve 'models/churn_pipeline.pkl' ile 'models/metrics.json' dosyalarını oluşturur:
```bash
  python src/train.py
```

### 3. API'yi Çalıştırma
```bash
  uvicorn src.app:app --reload
```

API çalıştıktan sonra servislere şu adreslerden ulaşabilirsiniz:

Swagger UI: http://127.0.0.1:8000/docs (Tüm parametreleri test etmek için)

Health Check: http://127.0.0.1:8000/health

### 4. Kullanıcı Arayüzünü Açma (Frontend)
Yeni bir terminal sekmesi açarak Streamlit uygulamasını başlatın:
```bash
  streamlit run src/frontend.py
```

## 🔌 API Kullanımı (Predict Endpoint)
Endpoint: POST /predict

### Örnek Request Body (JSON):
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
### Örnek Response (JSON):
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability": 0.8123
}
```

## 📝 Geliştirme Notları
  Veri setindeki TotalCharges sütunu numerik tipe çevrilmiştir.

  Gereksiz gürültüyü önlemek için customerID model girdisinden çıkarılmıştır.

  Kategorik değişkenler için OneHotEncoder, sayısal değişkenler için StandardScaler kullanılmıştır.

  Eğitim ve test verisi ayrılırken sınıf dengesizliğini korumak için train_test_split(..., stratify=y) uygulanmıştır.

