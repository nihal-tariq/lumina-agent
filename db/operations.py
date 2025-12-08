import psycopg
from psycopg.rows import tuple_row


def insert_university(database_url: str, uni_name: str, url: str, summary:str,  time_stamp=None):
    """
    Inserts a row into university table.
    time_stamp can be None → defaults to NOW().
    """
    insert_query = """
    INSERT INTO university (uni_name, url, summary, time_stamp)
    VALUES (%s, %s, %s, COALESCE(%s, NOW()))
    RETURNING id;
    """

    with psycopg.connect(database_url, row_factory=tuple_row) as conn:
        with conn.cursor() as cur:
            cur.execute(insert_query, (uni_name, url, summary, time_stamp))
            inserted_id = cur.fetchone()[0]

        conn.commit()

    return inserted_id
