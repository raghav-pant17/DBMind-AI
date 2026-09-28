"""Database connection management."""

from sqlalchemy import Engine, create_engine, event


# Creates a SQLAlchemy engine for the configured database URL.
def create_database_engine(database_url: str, schema_name: str | None = None) -> Engine:
    """Create a SQLAlchemy engine."""

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, future=True)
    if schema_name and not database_url.startswith("sqlite"):
        _attach_search_path(engine, schema_name)
    return engine


# Adds a connection hook that applies the configured database schema.
def _attach_search_path(engine: Engine, schema_name: str) -> None:
    """Set the PostgreSQL search path after each DBAPI connection opens."""

    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_connection, _connection_record) -> None:
        previous_autocommit = dbapi_connection.autocommit
        dbapi_connection.autocommit = True
        try:
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f'SET search_path TO "{schema_name}"')
        finally:
            dbapi_connection.autocommit = previous_autocommit
