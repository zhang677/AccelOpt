import requests
from supabase import create_client, Client
import os

# Refer to AccelOpt/templates/schema_v2.txt for the database schema. 

# --- CONFIGURATION (Replace these with your actual Supabase URL and remember to set your Service Role Key) ---
SUPABASE_URL = "http://localhost:8000"  # Kong port
# Use the SERVICE_ROLE_KEY to bypass Row Level Security (RLS)
SUPABASE_KEY = os.getenv("SERVICE_ROLE_KEY")

# 1. Initialize the Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_all_table_names():
    """
    Queries the root of the REST API to get the OpenAPI spec,
    which lists all available tables in the 'public' schema.
    """
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    # PostgREST returns the schema definition at the root URL
    response = requests.get(f"{SUPABASE_URL}/rest/v1/", headers=headers)
    
    if response.status_code == 200:
        definitions = response.json().get('definitions', {})
        return list(definitions.keys())
    else:
        print(f"Failed to fetch schema: {response.text}")
        return []

def query_all_tables():
    tables = get_all_table_names()
    
    if not tables:
        print("No tables found or error fetching schema.")
        return

    print(f"Found {len(tables)} tables. Starting data export...\n")

    for table in tables:
        try:
            # Query all rows from the current table
            # .execute() returns a response object with a .data attribute
            response = supabase.table(table).select("*").execute()
            
            data = response.data
            print(f"--- Table: {table} ({len(data)} rows) ---")
            
            # Print first 2 rows as a sample
            for row in data[:2]:
                print(row)
            print("\n")
            
        except Exception as e:
            print(f"Error querying table '{table}': {e}")

if __name__ == "__main__":
    query_all_tables()