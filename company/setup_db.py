import sqlite3

conn = sqlite3.connect("company.db")
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL,
    hire_date TEXT
);

DELETE FROM employees;

INSERT INTO employees VALUES
(1, 'Alice', 'Engineering', 95000, '2021-03-01'),
(2, 'Bob', 'Marketing', 72000, '2020-06-15'),
(3, 'Carol', 'Engineering', 105000, '2019-01-10'),
(4, 'Dave', 'HR', 68000, '2022-09-01');

CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    amount REAL,
    sale_date TEXT
);

DELETE FROM sales;

INSERT INTO sales VALUES
(1, 1, 15000, '2024-01-15'),
(2, 2, 8000, '2024-02-20'),
(3, 1, 22000, '2024-03-10');
""")
conn.commit()
conn.close()
print("Database ready: company.db")
