# 1. Hafif bir Python sürümü seçiyoruz
FROM python:3.10-slim

# 2. Çalışma klasörümüzü belirliyoruz
WORKDIR /app

# 3. Önce kütüphane listesini kopyalayıp kuruyoruz (Önbellek - Cache avantajı için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Projedeki tüm kodları ve verileri kapsülün içine kopyalıyoruz
COPY . .

# 5. API'nin çalışacağı portu dış dünyaya açıyoruz
EXPOSE 8000

# 6. Konteyner çalıştığında API'yi başlatan komut
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]