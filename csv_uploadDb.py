import os
import pandas as pd
import psycopg2
from psycopg2 import sql

# Database connection details
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "runtime"
DB_USER = "postgres"
DB_PASSWORD = "Welcom@123"

# Path to your CSV file
CSV_FILE_Path = r"C:\Users\seai_\Downloads\"
# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cursor = conn.cursor()

# Read the CSV file into a DataFrame
df = pd.read_csv(CSV_FILE_Path)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()  # Clean column names

print(f"?? Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")

# Generate table name from file name
dynamic_path = os.path.basename(CSV_FILE_Path).replace("\\", "_").replace("/", "_").replace(".", "_").replace("-", "_").lower()
table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in dynamic_path)

# Map pandas dtypes to PostgreSQL column types
def map_dtype(dtype, col_name):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        if "percent" in col_name.lower() or "%" in col_name.lower():
            return "FLOAT"
        return "DOUBLE PRECISION"
    return "TEXT"

columns = df.columns.tolist()
types = [map_dtype(df[col].dtype, col) for col in columns]

# Check if the table exists
cursor.execute(
    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);",
    (table_name,)
)
table_exists = cursor.fetchone()[0]

# Create table if it doesn't exist
if not table_exists:
    create_table_query = sql.SQL(
        "CREATE TABLE {} ({});"
    ).format(
        sql.Identifier(table_name),
        sql.SQL(', ').join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(data_type)) for col, data_type in zip(columns, types)
        )
    )
    cursor.execute(create_table_query)
    conn.commit()
    print(f"? Created new table: {table_name}")
else:
    print(f"?? Table {table_name} already exists. Checking columns...")

    # Check if columns exist; add any missing
    cursor.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s;",
        (table_name,)
    )
    existing_columns = [row[0] for row in cursor.fetchall()]
    for col, col_type in zip(columns, types):
        if col not in existing_columns:
            alter_table_query = sql.SQL(
                "ALTER TABLE {} ADD COLUMN {} {};"
            ).format(
                sql.Identifier(table_name),
                sql.Identifier(col),
                sql.SQL(col_type)
            )
            cursor.execute(alter_table_query)
            conn.commit()
            print(f"? Added missing column: {col} ({col_type})")

# Insert data into the table (no conflict handling)
insert_query = sql.SQL(
    "INSERT INTO {} ({}) VALUES ({});"
).format(
    sql.Identifier(table_name),
    sql.SQL(', ').join(sql.Identifier(col) for col in columns),
    sql.SQL(', ').join(sql.Placeholder() for _ in columns)
)

cursor.executemany(insert_query, df.values.tolist())
conn.commit()
print(f"? Inserted {df.shape[0]} rows into table: {table_name}")

# Close the connection
cursor.close()
conn.close()
print("?? Database connection closed.")