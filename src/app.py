from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from src.predict import get_prediction, load_model

# --- API YAPILANDIRMASI ---
app = FastAPI(title="Müşteri Kaybı (Churn) Tahmin API", version="1.0.0")

# Uygulama başlatıldığında modelin bir kez belleğe yüklenmesini sağlarız. 
# Bu sayede her tahminde dosyayı tekrar okumak zorunda kalmayız (Performans Optimizasyonu).
try:
    load_model() 
except Exception as e:
    print(f"Model yükleme hatası: {e}")

# --- VERİ DOĞRULAMA MODELİ Pydantic ---
class CustomerData(BaseModel):
    """
    API'ye gönderilecek müşteri verilerinin tipini ve sınırlarını belirleyen şema.
    Hatalı veri girişini (negatif ücret, yanlış cinsiyet vb.) daha en başında engeller.
    """

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1, description="0: Genç, 1: Yaşlı")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, description="Şirkette geçirilen ay sayısı")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

# --- ENDPOINT'LER  ---

@app.get("/", tags=["Genel"])
def read_root():
    """API'nin çalışıp çalışmadığını kontrol eden ana karşılama mesajı."""
    return {
        "status": "online",
        "message": "Customer Churn API aktif. Teknik dökümantasyon için lütfen /docs adresini ziyaret edin."
    }

@app.get("/health", tags=["Sistem"])
def health():
    """Sistemin ve modelin sağlık durumunu denetler."""
    model = load_model()
    return {"status": "ok" if model is not None else "model_not_loaded"}

@app.post("/predict", tags=["Tahmin"])
def predict_churn(customer: CustomerData):
    """
    Gelen müşteri verilerini alır, modele gönderir ve Churn (Ayrılma) ihtimalini döner.
    
    - **customer**: CustomerData şemasına uygun JSON verisi.
    - **return**: Ayrılma olasılığı (%) ve Risk durumu.
    """
    try:
        # Pydantic nesnesini sözlüğe çevirerek tahmin fonksiyonuna iletiyoruz
        result = get_prediction(customer.model_dump())
        return result
    except ValueError as ve:
        # Veri işleme sırasında oluşabilecek mantıksal hataları yakalar
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        # Beklenmedik tüm diğer hataları yakalar ve 400 döner
        raise HTTPException(status_code=400, detail=str(e))