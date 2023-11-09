import pymysql

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
cursor.execute("ALTER TABLE {} MODIFY COLUMN data_image BLOB(2000000)".format('0417_qian'))
cursor.execute("ALTER TABLE {} MODIFY COLUMN data_depth BLOB(2000000)".format('0417_qian'))

#打印数据表0417_qian所有数据列
cursor.execute("SELECT name FROM {}".format('0417_qian'))
file_names = [row[0] for row in cursor.fetchall()]
print(file_names)


# #打印数据表cabels_type所有数据列
# cursor.execute("SELECT name FROM {}".format('cabels_type'))
# rows = cursor.fetchall()
# print(rows)

# # 删除所有数据表
# for table in tables:
#     cursor.execute("DROP TABLE IF EXISTS {}".format(table[0]))
# # 提交更改并关闭连接
# connection.commit()

#在数据表里面添加线缆数据信息
sql = "CREATE TABLE IF NOT EXISTS {} (name varchar(255) PRIMARY KEY, waijing varchar(20), bilv varchar(20))".format('cabels_type')
cursor.execute(sql)
sql_waijing = 'INSERT INTO {} (name, waijing) VALUES (%s, %s) ON DUPLICATE KEY UPDATE waijing=VALUES(waijing)'.format('cabels_type')
sql_bilv = 'INSERT INTO {} (name, bilv) VALUES (%s, %s) ON DUPLICATE KEY UPDATE bilv=VALUES(bilv)'.format('cabels_type')
cursor.execute(sql_waijing, ('L05', 5))
cursor.execute(sql_bilv,('L05', 10))
cursor.execute(sql_waijing, ('L09', 5))
cursor.execute(sql_bilv,('L09', 10))
cursor.execute(sql_waijing, ('L10', 5))
cursor.execute(sql_bilv,('L10', 10))
cursor.execute(sql_waijing, ('L12', 5))
cursor.execute(sql_bilv,('L12', 10))
cursor.execute(sql_waijing, ('L18', 5))
cursor.execute(sql_bilv,('L18', 10))
connection.commit()

# 查询所有数据表
cursor.execute("SHOW TABLES")
tables = cursor.fetchall()

# 打印所有数据表
for table in tables:
    print(table[0])

#获取指定型号线缆信息
cursor.execute("SELECT * FROM cabels_type WHERE name=%s", ('L05',))
result = cursor.fetchone()
print(result)

# 关闭连接
cursor.close()
connection.close()
