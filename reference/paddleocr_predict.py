import time
from paddleocr import PaddleOCR
import cv2
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
import pymysql
import numpy as np

# 连接数据库
connection = pymysql.connect(host='127.0.0.1',
                             user='root',
                             password='123456',
                             database='buxianjianche')
cursor = connection.cursor()

ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = '../digital_pictures/4.jpg'
img = cv2.imread(img_path)
result = ocr.ocr(img, cls=True)
print(result)
if not result[0]:
    print(None)
for idx in range(len(result)):
    print(len(result))
    res = result[idx]
    for line in res:
        print(line)
        bbox = line[0]
        # 得到轮廓
        contour = np.array([bbox], dtype=np.int32)
        # 得到最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        text = line[1][0]
        cursor.execute("SELECT * FROM cabels_type WHERE name=%s", ('in',))
        result = cursor.fetchone()
        print(result == None)
        # self.lineEdit_waijing.setText(result[1])
        # self.lineEdit_bilv.setText(result[2])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        print(line[1][0])

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
