"""
HealthGuard DB — config.py
MySQL database configuration
"""

from urllib.parse import quote_plus

class Config:
    # Secret key for session management
    SECRET_KEY = 'healthguard_secret_2025'

    # ── MySQL Configuration ───────────────────────
    # Change these values to match your MySQL setup
    MYSQL_HOST     = 'localhost'
    MYSQL_USER     = 'root'
    MYSQL_PASSWORD = 'Rahul@2005'          # Enter your MySQL password here
    MYSQL_DB       = 'healthguard_flask'

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
        f"@{MYSQL_HOST}/{MYSQL_DB}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
