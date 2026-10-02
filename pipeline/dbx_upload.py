"""Write pandas frames into Delta tables of the sandbox schema, idempotently.

The job is small and daily, so rows go up as SQL — a MERGE whose source is an
inline VALUES table — through the Databricks SQL connector against the desk
apps' warehouse. No Spark, no volume, no file upload: a few thousand rows a day
is a handful of statements, and re-running a day (the SWE model re-reads its
last 14 days; EQ revises backcasts) updates rather than duplicates.

    w = DatabricksWriter.from_settings(settings)
    w.ensure_table("cat.schema.t", [("day", "DATE"), ("value", "DOUBLE")], comment="…")
    w.merge("cat.schema.t", df, keys=["day"])

`dialect` exists so the generated SQL can be exercised on DuckDB in tests
('duckdb'); production is always 'spark'.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.dbx")


# ─── SQL rendering ──────────────────────────────────────────────────────────────

def sql_literal(v, dialect: str = "spark") -> str:
    """One Python / pandas scalar as a SQL literal."""
    if v is None:
        return "NULL"
    if isinstance(v, float) and math.isnan(v):
        return "NULL"
    if v is pd.NaT:
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return "NULL"
        r = repr(float(v))
        return f"{r}D" if dialect == "spark" else r          # Spark: D suffix = DOUBLE literal (else DECIMAL)
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return "NULL"
        if v.tzinfo is not None:
            v = v.tz_convert("UTC").tz_localize(None)
        return f"TIMESTAMP '{v.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(v, dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return f"TIMESTAMP '{v.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(v, dt.date):
        return f"DATE '{v.isoformat()}'"
    if isinstance(v, np.datetime64):
        return sql_literal(pd.Timestamp(v), dialect)
    s = str(v).replace("\\", "\\\\").replace("'", "''") if dialect == "spark" else str(v).replace("'", "''")
    return f"'{s}'"


def _row_values(row: Sequence, dialect: str, types: Sequence[str | None]) -> str:
    """One VALUES tuple. A NULL is typed (CAST(NULL AS TIMESTAMP)) when the column
    type is known, so a chunk whose column is entirely NULL — a batch without a
    forecast — still has a typed column and MERGE never sees a void type."""
    parts = []
    for v, t in zip(row, types):
        lit = sql_literal(v, dialect)
        parts.append(f"CAST(NULL AS {t})" if (lit == "NULL" and t) else lit)
    return "(" + ", ".join(parts) + ")"


def _values_source(df: pd.DataFrame, dialect: str, types: dict[str, str] | None = None) -> str:
    cols = ", ".join(df.columns)
    col_types = [(types or {}).get(c) for c in df.columns]
    rows = ",\n    ".join(_row_values(r, dialect, col_types) for r in df.itertuples(index=False, name=None))
    if dialect == "spark":
        # Spark's documented inline-table form
        return f"SELECT * FROM VALUES\n    {rows}\n  AS v({cols})"
    return f"SELECT * FROM (VALUES\n    {rows}) AS v({cols})"


def merge_statement(table: str, df: pd.DataFrame, keys: Sequence[str], dialect: str = "spark",
                    types: dict[str, str] | None = None) -> str:
    """MERGE INTO table USING (inline VALUES) — null-safe on the keys, update the
    rest when matched, insert when not. `types` (column -> SQL type) types the
    NULL literals; the writers pass what ensure_table declared."""
    cols = list(df.columns)
    non_keys = [c for c in cols if c not in keys]
    eq = "<=>" if dialect == "spark" else "IS NOT DISTINCT FROM"
    on = " AND ".join(f"t.{k} {eq} s.{k}" for k in keys)
    update = ", ".join(f"{c} = s.{c}" for c in non_keys) if non_keys else None
    insert_cols = ", ".join(cols)
    insert_vals = ", ".join(f"s.{c}" for c in cols)
    stmt = f"MERGE INTO {table} t\nUSING (\n  {_values_source(df, dialect, types)}\n) s\nON {on}\n"
    if update:
        stmt += f"WHEN MATCHED THEN UPDATE SET {update}\n"
    stmt += f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    return stmt


def create_table_statement(table: str, columns: Sequence[tuple[str, str]], comment: str = "",
                           dialect: str = "spark") -> str:
    cols = ",\n  ".join(f"{n} {t}" for n, t in columns)
    stmt = f"CREATE TABLE IF NOT EXISTS {table} (\n  {cols}\n)"
    if dialect == "spark":
        stmt += " USING DELTA"
        if comment:
            stmt += f"\nCOMMENT '{comment.replace(chr(39), chr(39) * 2)}'"
    return stmt


def _chunks(df: pd.DataFrame, n: int) -> Iterable[pd.DataFrame]:
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n]


def prepare_frame(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Keep the last duplicate per key (MERGE refuses a source that matches one
    target row twice; NULL keys compare equal here as they do in the null-safe
    ON clause) and turn pandas NA into None."""
    out = df.drop_duplicates(subset=list(keys), keep="last").reset_index(drop=True)
    return out.astype(object).where(pd.notna(out), None)


# ─── Writer ─────────────────────────────────────────────────────────────────────

class DatabricksWriter:
    def __init__(self, host: str, http_path: str, token: str = "", auth_type: str = "",
                 tls_no_verify: bool = False, chunk_rows: int = 2000, dialect: str = "spark"):
        self.host, self.http_path, self.token, self.auth_type = host, http_path, token, auth_type
        self.tls_no_verify, self.chunk_rows, self.dialect = tls_no_verify, chunk_rows, dialect
        self._conn = None
        self._types: dict[str, dict[str, str]] = {}

    @classmethod
    def from_settings(cls, s) -> "DatabricksWriter":
        return cls(s.dbx_host, s.dbx_http_path, s.dbx_token, s.dbx_auth_type, s.dbx_tls_no_verify, s.merge_chunk_rows)

    # -- connection ----------------------------------------------------------
    def connect(self):
        if self._conn is not None:
            return self._conn
        from databricks import sql as dbsql
        # access_token is a positional parameter in connector 2.x (snow_obs has 2.0.2), so it is
        # always passed; auth_type (OAuth, Azure CLI, …) needs connector >= 3 and is ignored by 2.x.
        kwargs = {"server_hostname": self.host, "http_path": self.http_path, "access_token": self.token or None}
        if not self.token:
            if not self.auth_type:
                raise RuntimeError("No Databricks credentials: set DATABRICKS_TOKEN (PAT) in pipeline/.env "
                                   "(or DATABRICKS_AUTH_TYPE with databricks-sql-connector >= 3)")
            kwargs["auth_type"] = self.auth_type       # e.g. 'databricks-oauth' (browser login), 'azure-oauth'
            log.info("no PAT — auth_type=%s (requires databricks-sql-connector >= 3, snow_obs has %s)",
                     self.auth_type, getattr(dbsql, "__version__", "?"))
        if self.tls_no_verify:
            kwargs["_tls_no_verify"] = True
        log.info("connecting to %s %s", self.host, self.http_path)
        self._conn = dbsql.connect(**kwargs)
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def execute(self, statement: str, fetch: bool = False):
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(statement)
            if fetch:
                return cur.fetchall()
        return None

    def query_df(self, statement: str) -> pd.DataFrame:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(statement)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame([tuple(r) for r in rows], columns=cols)

    # -- DDL / DML ------------------------------------------------------------
    def ensure_table(self, table: str, columns: Sequence[tuple[str, str]], comment: str = "") -> None:
        self._types[table] = dict(columns)
        self.execute(create_table_statement(table, columns, comment, self.dialect))

    def merge(self, table: str, df: pd.DataFrame, keys: Sequence[str]) -> int:
        """MERGE the frame in chunks; returns the number of rows sent."""
        if df is None or df.empty:
            log.info("%s: nothing to write", table)
            return 0
        clean = prepare_frame(df, keys)
        n = 0
        for i, part in enumerate(_chunks(clean, self.chunk_rows), 1):
            self.execute(merge_statement(table, part, keys, self.dialect, self._types.get(table)))
            n += len(part)
            log.info("%s: chunk %d — %d rows merged (%d so far)", table, i, len(part), n)
        return n

    def prune(self, table: str, where: str) -> None:
        """DELETE rows matching `where` (Delta supports it) — keeps fast-growing tables bounded."""
        try:
            self.execute(f"DELETE FROM {table} WHERE {where}")
            log.info("%s: pruned rows WHERE %s", table, where)
        except Exception as e:
            log.warning("%s: prune failed (%s)", table, str(e)[:160])

    def table_max(self, table: str, column: str, group_by: str | None = None, where: str = "") -> pd.DataFrame | None:
        """MAX(column) [per group_by] or None when the table does not exist yet."""
        try:
            if group_by:
                return self.query_df(f"SELECT {group_by}, MAX({column}) AS mx FROM {table} {('WHERE ' + where) if where else ''} GROUP BY {group_by}")
            return self.query_df(f"SELECT MAX({column}) AS mx FROM {table} {('WHERE ' + where) if where else ''}")
        except Exception as e:  # table missing, no grant, …
            log.info("%s: cannot read MAX(%s) (%s) — treating as empty", table, column, str(e)[:120])
            return None


class DryRunWriter:
    """Stands in for DatabricksWriter when there are no credentials: keeps the
    frames, writes them to CSV in out_dir and logs the first statement."""

    def __init__(self, out_dir, chunk_rows: int = 2000):
        from pathlib import Path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = chunk_rows
        self.dialect = "spark"
        self.statements: list[str] = []
        self._types: dict[str, dict[str, str]] = {}

    def ensure_table(self, table, columns, comment=""):
        self._types[table] = dict(columns)
        self.statements.append(create_table_statement(table, columns, comment))
        log.info("[dry-run] %s", self.statements[-1].splitlines()[0])

    def merge(self, table, df, keys) -> int:
        if df is None or df.empty:
            log.info("[dry-run] %s: nothing to write", table)
            return 0
        clean = prepare_frame(df, keys)
        path = self.out_dir / f"{table.split('.')[-1]}_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        clean.to_csv(path, index=False, encoding="utf-8")
        first = next(_chunks(clean, self.chunk_rows))
        self.statements.append(merge_statement(table, first, keys, "spark", self._types.get(table)))
        (self.out_dir / f"{table.split('.')[-1]}_merge_example.sql").write_text(self.statements[-1], encoding="utf-8")
        log.info("[dry-run] %s: %d rows -> %s (first MERGE %d chars)", table, len(clean), path.name, len(self.statements[-1]))
        return len(clean)

    def table_max(self, *a, **k):
        return None

    def prune(self, table, where):
        log.info("[dry-run] DELETE FROM %s WHERE %s", table, where)

    def close(self):
        pass
