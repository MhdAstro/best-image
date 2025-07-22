# --- مرحله ۱: Builder ---
# در این مرحله، تمام وابستگی‌ها را در یک محیط موقت نصب می‌کنیم.
FROM python:3.11-slim as builder

# نصب ابزارهای مورد نیاز برای کامپایل برخی از پکیج‌ها
RUN apt-get update && apt-get install -y build-essential

# تنظیم پوشه کاری
WORKDIR /app

# کپی کردن فایل نیازمندی‌ها و نصب آن‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- مرحله ۲: Final ---
# در این مرحله، فقط فایل‌های نهایی و کدهای اجرایی را به یک ایمیج تمیز منتقل می‌کنیم.
FROM python:3.11-slim

# تنظیم پوشه کاری
WORKDIR /app

# کپی کردن پکیج‌های نصب شده از مرحله قبل (بدون ابزارهای اضافی)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# کپی کردن کد اصلی برنامه
COPY main.py .

# باز کردن پورت ۸۰۰۰ برای دسترسی به سرویس از بیرون کانتینر
EXPOSE 8000

# دستور نهایی برای اجرای سرور در حالت پروداکشن
# --host 0.0.0.0: برای اینکه سرور از خارج از کانتینر قابل دسترس باشد.
# --workers 2: اجرای ۲ پراسس برای مدیریت همزمان درخواست‌ها (این عدد را می‌توانید بر اساس CPU سرور تغییر دهید).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]