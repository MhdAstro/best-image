# --- STAGE 1: Builder ---
# در این مرحله، یک محیط بیلد تمیز ایجاد کرده و وابستگی‌ها را نصب می‌کنیم.
FROM python:3.9-slim AS builder

# ۱. تنظیم متغیرهای محیطی
# PYTHONDONTWRITEBYTECODE: جلوگیری از ساخت فایل‌های .pyc
# PYTHONUNBUFFERED: نمایش مستقیم لاگ‌ها در کنسول داکر
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ۲. نصب وابستگی‌های سیستمی (اگر نیاز بود)
# برای کتابخانه‌هایی مثل opencv-python-headless معمولاً نیازی نیست،
# اما اگر در آینده به کتابخانه‌ای با وابستگی سیستمی نیاز داشتی، اینجا اضافه کن.
# RUN apt-get update && apt-get install -y --no-install-recommends ...

# ۳. ایجاد یک محیط مجازی برای نصب پکیج‌ها
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ۴. کپی فایل نیازمندی‌ها و نصب پکیج‌ها
# این کار باعث می‌شه تا زمانی که requirements.txt تغییری نکرده، داکر از کش لایه‌ها استفاده کنه.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Runner ---
# در این مرحله، از ایمیج سبک پایتون استفاده کرده و فقط فایل‌های اجرایی و وابستگی‌های نصب‌شده را کپی می‌کنیم.
FROM python:3.9-slim

# ۱. ایجاد یک کاربر غیر روت برای افزایش امنیت
RUN addgroup --system app && adduser --system --group app
USER app

# ۲. تنظیم پوشه کاری
WORKDIR /home/app

# ۳. کپی محیط مجازی از مرحله Builder
COPY --from=builder /opt/venv /opt/venv

# ۴. کپی سورس کد برنامه
COPY main.py .

# ۵. تنظیم متغیرهای محیطی برای اجرای برنامه
ENV PATH="/opt/venv/bin:$PATH"
# تنظیماتی که در main.py استفاده شده‌اند را می‌توان از طریق متغیرهای محیطی اینجا مقداردهی کرد.
# به عنوان مثال:
# ENV SUBJECT_PROMINENCE_WEIGHT=0.5

# ۶. باز کردن پورت ۸۰۰۰ برای دسترسی به API
EXPOSE 8000

# ۷. دستور اجرای برنامه
# با استفاده از uvicorn سرور FastAPI را اجرا می‌کنیم و به هاست 0.0.0.0 متصل می‌شویم تا از خارج کانتینر قابل دسترس باشد.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]