from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)
image_path = '../data/input/images/1.jpg'
print(f'Testing Direct OCR on {image_path}')

# Resize the image if it's too large, but for now just run directly
ocr_result = ocr.predict(image_path)
for res in ocr_result:
    if res:
        print('OCR Output:', res['rec_texts'])
        print('Scores:', res['rec_scores'])
