import sqlite3
import os

def check_database_schema(db_path):
    """Check the schema of the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\nDatabase: {os.path.basename(db_path)}")
        print("=" * 50)
        
        if not tables:
            print("No tables found in the database.")
        else:
            print(f"Found {len(tables)} tables:")
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")
                print("-" * 30)
                
                # Get table info
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                if not columns:
                    print("  No columns found")
                else:
                    print("Columns:")
                    for col in columns:
                        print(f"  {col[1]} ({col[2]}) - {'PRIMARY KEY' if col[5] > 0 else ''}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"  Rows: {count:,}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    # Check all database files in the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    db_files = [f for f in os.listdir(data_dir) if f.endswith('.db')]
    
    if not db_files:
        print("No database files found in the data directory.")
    else:
        for db_file in db_files:
            db_path = os.path.join(data_dir, db_file)
            check_database_schema(db_path)
