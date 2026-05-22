# test_connection.py
import pyodbc
import pandas as pd

conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=NGUYEN_MINH;"
    "DATABASE=ProjectADY_StockDB;"
    "Trusted_Connection=yes;"
)

try:
    conn = pyodbc.connect(conn_str)
    print("✅ Kết nối OK!")
    
    # Xem cấu trúc cột thật của bảng
    df_cols = pd.read_sql("""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'daily_prices'
        ORDER BY ORDINAL_POSITION
    """, conn)
    print("\n📋 Cột trong bảng daily_prices:")
    print(df_cols.to_string())
    
    # Xem 3 dòng dữ liệu mẫu
    df_sample = pd.read_sql("SELECT TOP 3 * FROM dbo.daily_prices", conn)
    print("\n📊 Dữ liệu mẫu:")
    print(df_sample.to_string())
    
    conn.close()

except Exception as e:
    print("❌ Lỗi:", e)