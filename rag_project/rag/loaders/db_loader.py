"""
DB Loader — Retrieves data from a MySQL database.

Connects to the database specified in SOURCE_DB_* settings,
fetches all tables, and extracts all rows as text.
"""

import logging
import pymysql

from django.conf import settings

logger = logging.getLogger(__name__)


class DBLoader:
    """Extracts text data from a MySQL database."""

    def load(self) -> str:
        """
        Connect to the MySQL database, iterate over all tables,
        and retrieve all data. Each row is formatted as a string.

        Returns:
            Extracted text as a single string.
            
        Raises:
            Exception: If database connection or querying fails.
        """
        logger.info("Connecting to source database %s on %s", settings.SOURCE_DB_NAME, settings.SOURCE_DB_HOST)

        try:
            connection = pymysql.connect(
                host=settings.SOURCE_DB_HOST,
                user=settings.SOURCE_DB_USER,
                password=settings.SOURCE_DB_PASSWORD,
                database=settings.SOURCE_DB_NAME,
                port=int(settings.SOURCE_DB_PORT),
                cursorclass=pymysql.cursors.DictCursor
            )
        except Exception as e:
            logger.error("Failed to connect to source database: %s", e)
            raise ValueError(f"Database connection failed: {e}") from e

        text_parts = []
        
        try:
            with connection.cursor() as cursor:
                # Get all tables
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
                for table_dict in tables:
                    # The key for table name in SHOW TABLES depends on DB name, so we just get the first value
                    table_name = list(table_dict.values())[0]
                    logger.info("Loading data from table: %s", table_name)
                    
                    text_parts.append(f"--- Table: {table_name} ---")
                    
                    cursor.execute(f"SELECT * FROM `{table_name}`")
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        row_str = " | ".join([f"{k}: {v}" for k, v in row.items()])
                        text_parts.append(row_str)
                        
        except Exception as e:
            logger.error("Failed to query source database: %s", e)
            raise ValueError(f"Database query failed: {e}") from e
        finally:
            connection.close()
            
        final_text = "\n".join(text_parts)
        logger.info("DB loaded: %d characters", len(final_text))
        
        return final_text
