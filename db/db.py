import psycopg
from psycopg.rows import tuple_row


def create_tables(database_url: str):
    """
    Creates required tables in PostgreSQL if they do not already exist.
    Safe to run on every startup.
    """
    create_query = """
    CREATE TABLE IF NOT EXISTS university (
        id SERIAL PRIMARY KEY,
        uni_name TEXT NOT NULL,
        url TEXT NOT NULL,
        summary TEXT NOT NULL,
        time_stamp TIMESTAMPTZ DEFAULT NOW()
    );
    """

    with psycopg.connect(database_url, row_factory=tuple_row) as conn:
        with conn.cursor() as cur:
            cur.execute(create_query)
        conn.commit()
