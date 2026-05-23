import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=NGUYEN_MINH;"
    "DATABASE=ProjectADY_StockDB;"
    "Trusted_Connection=yes;"
)
cur = conn.cursor()
cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
tables = [r.TABLE_NAME for r in cur.fetchall()]
print("Tables:", tables)
for t in tables:
    cur.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{t}'")
    cols = [r.COLUMN_NAME for r in cur.fetchall()]
    print(f"  {t}: {cols}")
conn.close()
