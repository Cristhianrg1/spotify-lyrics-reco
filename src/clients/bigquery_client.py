from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter, LoadJobConfig
from google.oauth2 import service_account

from src.config.settings import get_settings


class BigQueryClient:
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        self.project_id = project_id or settings.gcp_project_id
        self.dataset = dataset or settings.bigquery_dataset

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID no está configurado en .env")
        if not self.dataset:
            raise ValueError("BIGQUERY_DATASET no está configurado en .env")

        sa_path = settings.google_application_credentials
        if not sa_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS no está configurado en .env")

        creds = service_account.Credentials.from_service_account_file(sa_path)

        self._client = bigquery.Client(
            project=self.project_id,
            credentials=creds,
        )

    # ---------- helpers ----------

    def table(self, table_name: str) -> str:
        """<project>.<dataset>.<table>"""
        return f"{self.project_id}.{self.dataset}.{table_name}"

    def _build_job_config(
        self,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Optional[QueryJobConfig]:
        if not params:
            return None

        query_params = []
        for name, value in params.items():
            if isinstance(value, bool):
                bq_type = "BOOL"
            elif isinstance(value, int):
                bq_type = "INT64"
            elif isinstance(value, float):
                bq_type = "FLOAT64"
            else:
                bq_type = "STRING"

            query_params.append(
                ScalarQueryParameter(name, bq_type, value),
            )

        return QueryJobConfig(query_parameters=query_params)

    # ---------- API pública ----------

    def query_df(
        self,
        sql: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        job_config = self._build_job_config(params)
        job = self._client.query(sql, job_config=job_config)
        result = job.result()
        return result.to_dataframe()

    def execute(
        self,
        sql: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        job_config = self._build_job_config(params)
        job = self._client.query(sql, job_config=job_config)
        job.result()

    def load_dataframe(
        self,
        table_name: str,
        df: pd.DataFrame,
        write_disposition: str = "WRITE_APPEND",
    ) -> None:
        if df.empty:
            return

        table_id = self.table(table_name)
        job_config = LoadJobConfig(write_disposition=write_disposition)

        job = self._client.load_table_from_dataframe(
            df,
            table_id,
            job_config=job_config,
        )
        job.result()
