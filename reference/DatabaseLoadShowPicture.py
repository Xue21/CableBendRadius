import pymysql
import numpy as np
import pickle
import cv2

# 连接数据库
connection = pymysql.connect(host='127.0.0.1',
                             user='root',
                             password='123456',
                             database='buxianjianche')
cursor = connection.cursor()
# 创建表格
sql = "CREATE TABLE IF NOT EXISTS {} (name varchar(255) PRIMARY KEY, data_image BLOB, data_depth BLOB)".format('0417_qian')
cursor.execute(sql)
# 增加data_image列的数据长度为2000000
cursor.execute("ALTER TABLE 0417_qian MODIFY COLUMN data_image BLOB(2000000)")
cursor.execute("ALTER TABLE 0417_qian MODIFY COLUMN data_depth BLOB(2000000)")
# 读取图像
img = cv2.imread('./img_wire/1.jpg')
#读取深度信息
depth_data = np.loadtxt('./img_wire/1.txt',delimiter=',')##loadtxt函数的默认分隔符是任意空格字符（包括空格、制表符和换行符）。如果您的文件使用不同的分隔符，请使用delimiter参数指定分隔符。
# 将图像转换为numpy数组
img_array = np.asarray(img)

# 将numpy数组序列化为二进制格式
img_bytes = pickle.dumps(img_array)
depth_bytes = pickle.dumps(depth_data)

# 插入二进制数据到数据库
sql = 'INSERT INTO {} (name, data_image) VALUES (%s, %s) ON DUPLICATE KEY UPDATE data_image=VALUES(data_image)'.format('0417_qian')
cursor.execute(sql, ('test_image', img_bytes))
connection.commit()
sql = 'INSERT INTO {} (name, data_depth) VALUES (%s, %s) ON DUPLICATE KEY UPDATE data_depth=VALUES(data_depth)'.format('0417_qian')
cursor.execute(sql, ('test_image', depth_bytes))
connection.commit()

# 从数据库中读取二进制数据并反序列化为numpy数组
cursor.execute("SELECT * FROM {} WHERE name=%s".format('0417_qian'), ('test_image',))
result = cursor.fetchone()
name = result[0]
img_bytes = result[1]
depth_bytes = result[2]
img_array = pickle.loads(img_bytes)
depth_array = pickle.loads(depth_bytes)

# 将numpy数组转换回图像
recovered_img = np.uint8(img_array)

# 显示图像
cv2.imshow('Recovered Image', recovered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(depth_array[10][0])
print(depth_array[13][0])

#查看所有已经保存的文件
cursor.execute("SELECT name FROM {}".format('0417_qian'))
result = cursor.fetchall()
file_names = [row[0] for row in result]
print(file_names)

# 关闭游标对象和数据库连接
cursor.close()
connection.close()

