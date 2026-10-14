import psycopg
import pymysql

print("--- Testing MariaDB 3307 ---")
for p in ['', 'root', 'admin', 'password', '1234', '123456', 'kavin', 'root123']:
    try:
        conn = pymysql.connect(host='localhost', port=3307, user='root', password=p)
        print(f"SUCCESS MariaDB 3307: password='{p}'")
        with conn.cursor() as cur:
            cur.execute('SHOW DATABASES')
            print('Databases:', cur.fetchall())
        conn.close()
        break
    except Exception as e:
        pass

print("--- Testing MySQL 3306 ---")
for p in ['', 'root', 'admin', 'password', '1234', '123456', 'kavin', 'root123']:
    try:
        conn = pymysql.connect(host='localhost', port=3306, user='root', password=p)
        print(f"SUCCESS MySQL 3306: password='{p}'")
        with conn.cursor() as cur:
            cur.execute('SHOW DATABASES')
            print('Databases:', cur.fetchall())
        conn.close()
        break
    except Exception as e:
        pass
