import asyncio
import aiohttp
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from rembg import remove
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from starlette.responses import Response
import logging
import uuid
import os

# Settings and other setup remain the same...
# --------------------------------------------------
# 1. مدیریت تنظیمات
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
app = FastAPI(title="Smart Book Image Processor API [v9.0 - Robust Straightening]")

class ImageRequest(BaseModel):
    image_urls: List[HttpUrl]

# --------------------------------------------------
# 3. توابع پردازشی (با تابع straighten_and_crop اصلاح شده)
# --------------------------------------------------

def calculate_image_score(image_data: bytes) -> Optional[Dict[str, Any]]:
    # This function remains the same as the last version
    try:
        image_pil = Image.open(BytesIO(image_data)).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w, _ = image_cv.shape
        image_area = h * w
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
    except Exception as e:
        logger.error(f"Error during score calculation: {e}", exc_info=True)
        return None

def straighten_and_crop(image_cv: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
    """
    تابع اصلاح شده با رفع خطای np.int0
    """
    try:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # ##<-- راه حل: استفاده از .astype(int) به جای np.int0 -->##
        # این روش استاندارد و مدرن برای تبدیل نوع داده در NumPy است
        box = box.astype(int)

        sorted_rect = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        sorted_rect[0] = box[np.argmin(s)]
        sorted_rect[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        sorted_rect[1] = box[np.argmin(diff)]
        sorted_rect[3] = box[np.argmax(diff)]
        
        (tl, tr, br, bl) = sorted_rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth == 0 or maxHeight == 0: return None

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(sorted_rect, dst)
        warped = cv2.warpPerspective(image_cv, M, (maxWidth, maxHeight))
        logger.info("Image successfully straightened using minAreaRect.")
        return warped
    except Exception as e:
        logger.error(f"Error during robust straightening: {e}", exc_info=True)
        return None

def process_background(image_data: bytes) -> Optional[bytes]:
    # This function remains the same
    try:
        output_bytes = remove(image_data)
        if not output_bytes: return None
        image_no_bg_pil = Image.open(BytesIO(output_bytes))
        if not image_no_bg_pil.getbbox(): return None
        white_bg = Image.new("RGBA", image_no_bg_pil.size, "WHITE")
        white_bg.paste(image_no_bg_pil, (0, 0), image_no_bg_pil)
        final_image_pil = white_bg.convert("RGB")
        output_buffer = BytesIO()
        final_image_pil.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error during background processing: {e}", exc_info=True)
        return None

# The main endpoint remains the same as the last corrected version
@app.post("/v1/process-book/")
async def process_book_images(request: ImageRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"--- New Request [{request_id}] ---")
    fetched_tasks = []
    async with aiohttp.ClientSession() as session:
        fetch_coroutines = [session.get(str(url), timeout=settings.REQUEST_TIMEOUT) for url in request.image_urls]
        responses = await asyncio.gather(*fetch_coroutines, return_exceptions=True)
        for i, res in enumerate(responses):
            if isinstance(res, Exception) or res.status != 200:
                logger.warning(f"Failed to fetch URL {request.image_urls[i]}")
            else:
                fetched_tasks.append(await res.read())
    if not fetched_tasks:
        raise HTTPException(status_code=400, detail="Could not fetch any valid images.")
    loop = asyncio.get_event_loop()
    score_results = await asyncio.gather(*[loop.run_in_executor(None, calculate_image_score, data) for data in fetched_tasks])
    candidates = []
    for i, result in enumerate(score_results):
        if result:
            candidates.append({"score": result["score"], "contour": result["contour"], "image_data": fetched_tasks[i]})
    if not candidates:
        raise HTTPException(status_code=400, detail="No images met the minimum scoring criteria.")
    valid_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    logger.info(f"Scoring complete. {len(valid_candidates)} valid candidates found.")
    top_candidates = valid_candidates[:settings.TOP_CANDIDATES_COUNT]
    final_image_bytes = None
    selection_status = "FALLBACK"
    final_score = top_candidates[0]['score']
    for i, candidate in enumerate(top_candidates):
        logger.info(f"Attempting to process Candidate #{i+1} (Score: {candidate['score']:.2f})")
        try:
            original_cv = cv2.cvtColor(np.array(Image.open(BytesIO(candidate['image_data']))), cv2.COLOR_RGB2BGR)
            straightened_cv = await loop.run_in_executor(None, straighten_and_crop, original_cv, candidate['contour'])
            if straightened_cv is None:
                logger.warning(f"Skipping candidate #{i+1} as it could not be straightened.")
                continue
            is_success, buffer = cv2.imencode(".png", straightened_cv)
            if not is_success: continue
            straightened_image_data = buffer.tobytes()
            processed_bytes = await loop.run_in_executor(None, process_background, straightened_image_data)
            if processed_bytes:
                final_image_bytes = processed_bytes
                selection_status = "SUCCESS"
                final_score = candidate['score']
                logger.info(f"SUCCESS! Candidate #{i+1} was fully processed.")
                break
        except Exception as e:
            logger.error(f"Critical error in processing loop for candidate #{i+1}: {e}", exc_info=True)
            continue
    if final_image_bytes is None:
        logger.warning(f"FALLBACK! No candidate was suitable for full processing.")
        final_image_bytes = top_candidates[0]['image_data']
    headers = {"X-Selection-Status": selection_status, "X-Final-Score": f"{final_score:.2f}", "Content-Type": "image/jpeg"}
    logger.info(f"--- Request [{request_id}] Finished --- Status: {selection_status}, Final Score: {final_score:.2f}")
    return Response(content=final_image_bytes, headers=headers)