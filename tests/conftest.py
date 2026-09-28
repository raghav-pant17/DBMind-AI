"""Shared test fixtures."""

from pathlib import Path

import pytest
from sqlalchemy import text

from dbmind_ai.database.connection import create_database_engine


@pytest.fixture
def demo_database_url(tmp_path: Path) -> str:
    """Create a temporary SQLite database used only by tests."""

    database_url = f"sqlite:///{tmp_path / 'demo.db'}"
    engine = create_database_engine(database_url)
    statements = [
        "CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name TEXT NOT NULL)",
        """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    region_id INTEGER NOT NULL,
    FOREIGN KEY(region_id) REFERENCES regions(region_id)
)
""",
        """
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    discount_amount REAL NOT NULL DEFAULT 0,
    refund_amount REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
)
""",
        "INSERT INTO regions VALUES (1, 'North America'), (2, 'APAC'), (3, 'EMEA')",
        """
INSERT INTO customers VALUES
(1, 'Acme Corp', 1),
(2, 'Blue Ocean', 2),
(3, 'Cobalt Systems', 3),
(4, 'Delta Retail', 1)
""",
        """
INSERT INTO orders VALUES
(1, 1, 120000, 5000, 0, 'PAID', '2025-02-14'),
(2, 2, 220000, 10000, 0, 'PAID', '2025-04-18'),
(3, 3, 175000, 0, 0, 'PAID', '2025-06-22'),
(4, 4, 90000, 0, 0, 'CANCELLED', '2025-07-03'),
(5, 2, 80000, 0, 5000, 'REFUNDED', '2025-08-11'),
(6, 1, 140000, 7000, 0, 'PAID', '2024-03-12')
""",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
    engine.dispose()
    return database_url
