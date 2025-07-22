# --- STAGE 1: Builder ---
# در این مرحله، یک محیط بیلد تمیز ایجاد کرده و وابستگی‌ها را نصب می‌کنیم.
FROM python:3.9-slim AS builder

# تنظیم متغیرهای محیطی برای بهینه‌سازی
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ایجاد یک محیط مجازی برای نصب پکیج‌ها
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# کپی فایل نیازمندی‌ها و نصب پکیج‌ها با استفاده از کش لایه‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Runner ---
# در این مرحله، از ایمیج سبک پایتون استفاده کرده و فقط فایل‌های اجرایی و وابستگی‌های نصب‌شده را کپی می‌کنیم.
FROM python:3.9-slim

# تنظیم پوشه کاری
WORKDIR /home/app

# کپی محیط مجازی از مرحله Builder
COPY --from=builder /opt/venv /opt/venv

# کپی سورس کد برنامه
COPY main.py .

# ایجاد کاربر و گروه غیر-روت برای افزایش امنیت
RUN addgroup --system app && adduser --system --group app

# تغییر مالکیت فایل‌های برنامه و محیط مجازی به کاربر جدید
# این خط کلیدی، مشکل دسترسی را حل می‌کند
RUN chown -R app:app /home/app /opt/venv

# سوییچ به کاربر غیر-روت
USER app

# افزودن محیط مجازی به PATH سیستم
ENV PATH="/opt/venv/bin:$PATH"

# باز کردن پورت ۸۰۰۰ برای دسترسی به API
EXPOSE 8000

# دستور اجرای برنامه با uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]