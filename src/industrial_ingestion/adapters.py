"""Source adapters that translate vendor/source payloads into canonical records."""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from industrial_ingestion.contracts import AdapterBatch, CanonicalIngestionRecord, SensorReference, SourceType


def _sensor_ref(raw: dict[str, Any]) -> SensorReference:
    return SensorReference(
        sensor_id=raw.get("sensor_id"),
        sensor_external_ref=raw.get("sensor_external_ref") or raw.get("sensorExternalRef") or raw.get("node_id"),
        site_slug=raw.get("site_slug"),
        asset_slug=raw.get("asset_slug"),
        component_slug=raw.get("component_slug"),
        sensor_slug=raw.get("sensor_slug"),
    )


def _canonical_record(raw: dict[str, Any]) -> CanonicalIngestionRecord:
    return CanonicalIngestionRecord(
        kind=raw.get("kind", "scalar"),
        observed_at=raw.get("observed_at") or raw.get("timestamp") or raw.get("ts"),
        sensor=_sensor_ref(raw),
        source_record_id=raw.get("source_record_id") or raw.get("id") or raw.get("sequence"),
        source_timezone=raw.get("source_timezone") or raw.get("timezone"),
        metric=raw.get("metric"),
        value=raw.get("value"),
        unit=raw.get("unit"),
        quality=raw.get("quality", "good"),
        sampling_rate_hz=raw.get("sampling_rate_hz") or raw.get("sample_rate_hz"),
        samples=raw.get("samples"),
        sample_count=raw.get("sample_count"),
        storage_uri=raw.get("storage_uri"),
        sha256=raw.get("sha256"),
        metadata=raw.get("metadata") or {},
    )


class IngestionAdapter(ABC):
    source_type: SourceType

    @abstractmethod
    def parse(self, payload: Any, *, source_name: str, **options: Any) -> AdapterBatch:
        raise NotImplementedError


class TabularAdapter(IngestionAdapter):
    def _from_frame(self, frame: pd.DataFrame, *, source_name: str, source_uri: str | None) -> AdapterBatch:
        records = [_canonical_record(row.dropna().to_dict()) for _, row in frame.iterrows()]
        return AdapterBatch(
            source_type=self.source_type,
            source_name=source_name,
            records=records,
            source_uri=source_uri,
            provenance={"adapter": self.__class__.__name__},
        )


class CsvAdapter(TabularAdapter):
    source_type: SourceType = "csv"

    def parse(self, payload: bytes | str, *, source_name: str, **options: Any) -> AdapterBatch:
        source_uri = options.get("source_uri")
        if isinstance(payload, bytes):
            payload = payload.decode(options.get("encoding", "utf-8"))
        frame = pd.read_csv(io.StringIO(payload))
        return self._from_frame(frame, source_name=source_name, source_uri=source_uri)


class ParquetAdapter(TabularAdapter):
    source_type: SourceType = "parquet"

    def parse(self, payload: bytes | str, *, source_name: str, **options: Any) -> AdapterBatch:
        source_uri = options.get("source_uri")
        if isinstance(payload, bytes):
            frame = pd.read_parquet(io.BytesIO(payload))
        else:
            frame = pd.read_parquet(payload)
        return self._from_frame(frame, source_name=source_name, source_uri=source_uri)


class RestAdapter(IngestionAdapter):
    source_type: SourceType = "rest"

    def parse(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        source_name: str,
        **options: Any,
    ) -> AdapterBatch:
        records_raw = payload.get("records", [payload]) if isinstance(payload, dict) else payload
        return AdapterBatch(
            source_type=self.source_type,
            source_name=payload.get("source_name", source_name) if isinstance(payload, dict) else source_name,
            records=[_canonical_record(raw) for raw in records_raw],
            source_uri=options.get("source_uri"),
            batch_idempotency_key=payload.get("batch_idempotency_key") if isinstance(payload, dict) else None,
            provenance={"adapter": self.__class__.__name__},
        )


class MqttAdapter(RestAdapter):
    source_type: SourceType = "mqtt"

    def parse(self, payload: bytes | str | dict[str, Any], *, source_name: str, **options: Any) -> AdapterBatch:
        if isinstance(payload, bytes):
            payload = payload.decode(options.get("encoding", "utf-8"))
        if isinstance(payload, str):
            payload = json.loads(payload)
        batch = super().parse(payload, source_name=source_name, **options)
        batch.source_type = self.source_type
        batch.provenance["topic"] = options.get("topic")
        return batch


class OpcUaAdapter(IngestionAdapter):
    source_type: SourceType = "opcua"

    def parse(self, payload: dict[str, Any], *, source_name: str, **options: Any) -> AdapterBatch:
        nodes = payload.get("nodes", payload.get("records", []))
        records = []
        for node in nodes:
            records.append(
                _canonical_record(
                    {
                        "node_id": node.get("node_id") or node.get("nodeId"),
                        "observed_at": node.get("server_timestamp") or node.get("timestamp"),
                        "metric": node.get("metric") or node.get("browse_name") or node.get("browseName"),
                        "value": node.get("value"),
                        "unit": node.get("unit"),
                        "quality": node.get("quality", "good"),
                        "source_record_id": node.get("source_record_id") or node.get("sequence"),
                    }
                )
            )
        return AdapterBatch(
            source_type=self.source_type,
            source_name=payload.get("source_name", source_name),
            records=records,
            provenance={"adapter": self.__class__.__name__, "endpoint": payload.get("endpoint")},
        )


class AbbAdapter(IngestionAdapter):
    source_type: SourceType = "abb"

    def parse(self, payload: dict[str, Any], *, source_name: str, **options: Any) -> AdapterBatch:
        measurements = payload.get("measurements", payload.get("records", []))
        records = []
        for measurement in measurements:
            records.append(
                _canonical_record(
                    {
                        "sensor_external_ref": measurement.get("sensorExternalRef")
                        or measurement.get("sensor_external_ref"),
                        "observed_at": measurement.get("observedAt") or measurement.get("observed_at"),
                        "metric": measurement.get("metric"),
                        "value": measurement.get("value"),
                        "unit": measurement.get("unit"),
                        "quality": measurement.get("quality", "good"),
                        "source_record_id": measurement.get("id") or measurement.get("source_record_id"),
                        "metadata": {"vendor": "abb", **(measurement.get("metadata") or {})},
                    }
                )
            )
        return AdapterBatch(
            source_type=self.source_type,
            source_name=payload.get("source_name", source_name),
            records=records,
            provenance={"adapter": self.__class__.__name__, "vendor": "abb"},
        )


class ReplayAdapter(IngestionAdapter):
    source_type: SourceType = "replay"

    def parse(self, payload: list[CanonicalIngestionRecord], *, source_name: str, **options: Any) -> AdapterBatch:
        return AdapterBatch(
            source_type=self.source_type,
            source_name=source_name,
            records=payload,
            source_uri=options.get("source_uri"),
            provenance={"adapter": self.__class__.__name__, "replay_of_batch_id": options.get("replay_of_batch_id")},
        )


ADAPTERS: dict[SourceType, IngestionAdapter] = {
    "csv": CsvAdapter(),
    "parquet": ParquetAdapter(),
    "rest": RestAdapter(),
    "mqtt": MqttAdapter(),
    "opcua": OpcUaAdapter(),
    "abb": AbbAdapter(),
    "replay": ReplayAdapter(),
}
