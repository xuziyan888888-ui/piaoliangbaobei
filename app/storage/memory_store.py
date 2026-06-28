import json
import sqlite3
import threading
from pathlib import Path

from app.models.job import JobRecord
from app.models.pipeline import (
    CandidateRecord,
    GenerationControlBundle,
    PreprocessResult,
    ReferenceParseResult,
)


class SQLiteStore:
    def __init__(self, db_path: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or str(data_dir / "piaoliangbaobei.db")
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preprocess_results (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_parse_results (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS control_bundles (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_job(self, job: JobRecord) -> None:
        payload = json.dumps(job.model_dump(mode="json"), ensure_ascii=False)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (job_id, payload_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(job_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (job.job_id, payload),
                )
                conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

        if row is None:
            return None

        payload = json.loads(row["payload_json"])
        return JobRecord.model_validate(payload)

    def save_candidates(self, job_id: str, candidates: list[CandidateRecord]) -> None:
        payloads = [
            (
                candidate.candidate_id,
                job_id,
                json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False),
            )
            for candidate in candidates
        ]

        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
                conn.executemany(
                    """
                    INSERT INTO candidates (candidate_id, job_id, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    payloads,
                )
                conn.commit()

    def get_candidates(self, job_id: str) -> list[CandidateRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM candidates
                WHERE job_id = ?
                ORDER BY created_at ASC, candidate_id ASC
                """,
                (job_id,),
            ).fetchall()

        return [
            CandidateRecord.model_validate(json.loads(row["payload_json"]))
            for row in rows
        ]

    def save_preprocess_result(self, job_id: str, result: PreprocessResult) -> None:
        self._save_single_payload("preprocess_results", job_id, result.model_dump(mode="json"))

    def get_preprocess_result(self, job_id: str) -> PreprocessResult | None:
        payload = self._get_single_payload("preprocess_results", job_id)
        if payload is None:
            return None
        return PreprocessResult.model_validate(payload)

    def save_reference_parse_result(self, job_id: str, result: ReferenceParseResult) -> None:
        self._save_single_payload("reference_parse_results", job_id, result.model_dump(mode="json"))

    def get_reference_parse_result(self, job_id: str) -> ReferenceParseResult | None:
        payload = self._get_single_payload("reference_parse_results", job_id)
        if payload is None:
            return None
        return ReferenceParseResult.model_validate(payload)

    def save_control_bundle(self, job_id: str, bundle: GenerationControlBundle) -> None:
        self._save_single_payload("control_bundles", job_id, bundle.model_dump(mode="json"))

    def get_control_bundle(self, job_id: str) -> GenerationControlBundle | None:
        payload = self._get_single_payload("control_bundles", job_id)
        if payload is None:
            return None
        return GenerationControlBundle.model_validate(payload)

    def _save_single_payload(self, table: str, job_id: str, payload_dict: dict) -> None:
        payload = json.dumps(payload_dict, ensure_ascii=False)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {table} (job_id, payload_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(job_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (job_id, payload),
                )
                conn.commit()

    def _get_single_payload(self, table: str, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload_json FROM {table} WHERE job_id = ?",
                (job_id,),
            ).fetchone()

        if row is None:
            return None
        return json.loads(row["payload_json"])


store = SQLiteStore()
