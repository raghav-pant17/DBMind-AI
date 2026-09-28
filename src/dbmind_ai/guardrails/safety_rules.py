"""SQL safety rules."""

FORBIDDEN_EXPRESSIONS = {
    "Alter",
    "Create",
    "Delete",
    "Drop",
    "Insert",
    "Merge",
    "Truncate",
    "Update",
}

FORBIDDEN_KEYWORDS = {
    "alter",
    "create",
    "delete",
    "drop",
    "insert",
    "merge",
    "truncate",
    "update",
}
