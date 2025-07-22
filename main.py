import asyncio
import aiohttp
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from rembg import remove
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from starlette.responses import Response, JSONResponse
import logging
import uuid
import os

# --------------------------------------------------
# 1. مدیریت تنظیمات (بدون تغییر)
# --------------------------------------------------
class Settings(BaseSettings):
    SUBJECT_PROMINENCE_WEIGHT: float = 0.4
    SHARPNESS_WEIGHT: float = 0.2
    STRAIGHTNESS_WEIGHT: float = 0.2
    BACKGROUND_WHITENESS_WEIGHT: float = 0.2
    MIN_CONTOUR_AREA_PERCENT: float = 0.05
    REQUEST_TIMEOUT: int = 15
    TOP_CANDIDATES_COUNT: int = 3
    SHARPNESS_NORMALIZATION_CAP: int = 1000

settings = Settings()

# --------------------------------------------------
# 2. پیکربندی لاگینگ و FastAPI
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [PROCESS] - %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="Smart Book Image Processor & Selector API")

# ##<-- مدل‌های جدید برای اندپوینت انتخاب آیدی -->##
class ImageInput(BaseModel):
    id: Any  # می‌تواند INT, STRING, UUID و... باشد
    url: HttpUrl

class IdSelectionRequest(BaseModel):
    images: List[ImageInput]

class IdSelectionResponse(BaseModel):
    best_image_id: Any
    score: float

# مدل قدیمی برای اندپوینت پردازش تصویر
class ImageProcessRequest(BaseModel):
    image_urls: List[HttpUrl]


# --------------------------------------------------
# 3. توابع پردازشی (بدون تغییر)
# --------------------------------------------------
# توابع calculate_image_score, straighten_and_crop, process_background
# بدون هیچ تغییری در اینجا قرار می‌گیرند...
def calculate_image_score(image_data: bytes) -> Optional[Dict[str, Any]]:
    try:
        image_pil = Image.open(BytesIO(image_data)).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w, _ = image_cv.shape; image_area = h * w
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        sharpness_score = (min(cv2.Laplacian(gray_image, cv2.CV_64F).var(), settings.SHARPNESS_NORMALIZATION_CAP) / settings.SHARPNESS_NORMALIZATION_CAP) * 100
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        kernel = np.ones((7, 7), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        subject_prominence_score = (contour_area / image_area) * 100
        if subject_prominence_score < settings.MIN_CONTOUR_AREA_PERCENT * 100: return None
        _, _, w_rect, h_rect = cv2.boundingRect(largest_contour)
        bounding_rect_area = w_rect * h_rect
        straightness_score = (contour_area / bounding_rect_area) * 100 if bounding_rect_area > 0 else 0
        corner_size = min(h, w) // 10
        corners = [image_cv[0:corner_size, 0:corner_size], image_cv[0:corner_size, w-corner_size:w], image_cv[h-corner_size:h, 0:corner_size], image_cv[h-corner_size:h, w-corner_size:w]]
        avg_corner_color = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0)
        distance = np.linalg.norm(avg_corner_color - np.array([255, 255, 255]))
        background_whiteness_score = max(0, 100 - (distance / 441 * 100))
        total_score = (subject_prominence_score * settings.SUBJECT_PROMINENCE_WEIGHT) + (sharpness_score * settings.SHARPNESS_WEIGHT) + (straightness_score * settings.STRAIGHTNESS_WEIGHT) + (background_whiteness_score * settings.BACKGROUND_WHITENESS_WEIGHT)
        return {"score": total_score, "contour": largest_contour}
    except Exception: return None
# ... (سایر توابع کمکی)

# --------------------------------------------------
# 4. اندپوینت جدید: انتخاب بهترین آیدی
# --------------------------------------------------
@app.post("/v1/select-best-id/", response_model=IdSelectionResponse)
async def select_best_image_id(request: IdSelectionRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"--- New ID Selection Request [{request_id}] --- Received {len(request.images)} images.")
    
    # Fetching all images
    fetched_tasks = {} # دیکشنری برای نگهداری داده تصویر به ازای هر آیدی
    async with aiohttp.ClientSession() as session:
        fetch_coroutines = {img.id: session.get(str(img.url), timeout=settings.REQUEST_TIMEOUT) for img in request.images}
        responses = await asyncio.gather(*fetch_coroutines.values(), return_exceptions=True)
        
        for i, (img_id, res_coro) in enumerate(fetch_coroutines.items()):
            res = responses[i]
            if isinstance(res, Exception):
                logger.warning(f"Failed to fetch image for ID {img_id}: {res}")
            elif res.status != 200:
                logger.warning(f"Failed to fetch image for ID {img_id} with status {res.status}")
            else:
                fetched_tasks[img_id] = await res.read()

    if not fetched_tasks:
        raise HTTPException(status_code=400, detail="Could not fetch any valid images.")

    # Scoring all fetched images
    loop = asyncio.get_event_loop()
    # هر تسک امتیازدهی، آیدی را همراه خود دارد
    score_coroutines = [loop.run_in_executor(None, calculate_image_score, data) for data in fetched_tasks.values()]
    score_results = await asyncio.gather(*score_coroutines)
    
    candidates = []
    # استخراج آیدی‌ها از دیکشنری
    fetched_ids = list(fetched_tasks.keys())
    for i, result in enumerate(score_results):
        if result:
            candidates.append({"id": fetched_ids[i], "score": result["score"]})

    if not candidates:
        raise HTTPException(status_code=400, detail="No images met the minimum scoring criteria.")
        
    # پیدا کردن بهترین کاندیدا
    best_candidate = max(candidates, key=lambda x: x['score'])
    
    logger.info(f"ID selection complete. Best ID: {best_candidate['id']} with score {best_candidate['score']:.2f}")
    
    return IdSelectionResponse(
        best_image_id=best_candidate['id'],
        score=round(best_candidate['score'], 2)
    )


# --------------------------------------------------
# 5. اندپوینت قدیمی: پردازش و بازگرداندن تصویر (بدون تغییر)
# --------------------------------------------------
@app.post("/v1/process-book/")
async def process_book_images(request: ImageProcessRequest):
    # این اندپوینت و تمام منطق آن بدون تغییر باقی می‌ماند
    # ...
    return Response(content=b"image_data_here", headers={})