# Archived Core Modules

These files have been archived as part of the codebase consolidation.

## Archived Files

| File | Reason | Replacement |
|------|--------|-------------|
| `data_manager.py` | SQLite-only legacy version | Use `core/data_manager_postgres.py` |
| `data_manager_sqlite_backup.py` | Backup of legacy version | Use `core/data_manager_postgres.py` |

## Migration Notes

The canonical data manager is now `core/data_manager_postgres.py`, which:
- Automatically detects PostgreSQL if `DATABASE_URL` is set
- Falls back to SQLite otherwise
- Provides an identical interface to the legacy version

If you have code importing from `core.data_manager`, update to:

```python
from core.data_manager_postgres import DataManager
```

*Archived: 2026-02-11*
