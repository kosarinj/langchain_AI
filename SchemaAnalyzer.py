import mysql.connector

class SchemaAnalyzer:
    def __init__(self, host, user, password, database, port):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        self.cursor = self.connection.cursor()
        self.schema = {}

    def analyze_schema(self):
        self.cursor.execute("select distinct CONCAT(table_schema, '.', table_name) from information_schema.columns")
        tables = self.cursor.fetchall()

        for table in tables:
            table_name = table[0]
            self.schema[table_name] = {
                "columns": [],
                "foreign_keys": []
            }
            self._analyze_columns(table_name)
            self._analyze_foreign_keys(table_name)

    def _analyze_columns(self, table_name):
        self.cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = self.cursor.fetchall()
        for column in columns:
            self.schema[table_name]["columns"].append(column[0])

    def _analyze_foreign_keys(self, table_name):
        self.cursor.execute(f"""
            SELECT
                column_name,
                referenced_table_name,
                referenced_column_name
            FROM
                information_schema.key_column_usage
            WHERE
                table_name = '{table_name}' AND
                referenced_table_name IS NOT NULL
        """)
        foreign_keys = self.cursor.fetchall()
        for fk in foreign_keys:
            self.schema[table_name]["foreign_keys"].append({
                "column": fk[0],
                "referenced_table": fk[1],
                "referenced_column": fk[2]
            })

    def get_schema(self):
        return self.schema