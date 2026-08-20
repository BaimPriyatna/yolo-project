from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
result = ocr.predict('../data/input/images/2.jpg')

for res in result:
    res.print()
    print("Texts:", res['rec_texts'])
    print("Scores:", res['rec_scores'])