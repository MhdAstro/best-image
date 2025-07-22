# استفاده از یک ایمیج پایه کامل‌تر پایتون
FROM python:3.11

# نصب وابستگی‌های سیستمی مورد نیاز برای OpenCV و سایر کتابخانه‌ها
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# تنظیم پوشه کاری
WORKDIR /app

# کپی کردن فایل نیازمندی‌ها و نصب آن‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کد اصلی برنامه
COPY main.py .

# باز کردن پورت ۸۰۰۰
EXPOSE 8000

# دستور نهایی برای اجرای سرور
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]