"""Industrial ingestion service pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from industrial_ingestion.adapters import ADAPTERS
from industrial_ingestion.contracts import (
    AdapterBatch,
    CanonicalIngestionRecord,
    IngestionFailureReceipt,
    IngestionReceipt,
    SourceRegistration,
    SourceType,
)
from industrial_ingestion.units import normalize_unit_value
from industrial_ingestion.waveform_store import LocalWaveformStore
from platform_core.contracts import MachineReadingCreate
from platform_core.models import MachineReading, Organization, Sensor, WaveformRecord
from platform_core.repositories import PlatformRepository


class IngestionError(ValueError):
    pass


class SensorResolutionError(IngestionError):
    pass


class IngestionService:
    def __init__(self, session: Session, waveform_store: LocalWaveformStore | None = None):
        self.repo = PlatformRepository(session)
        self.waveform_store = waveform_store or LocalWaveformStore()

    def register_source(self, registration: SourceRegistration):
        if self.repo.session.get(Organization, registration.organization_id) is None:
            raise IngestionError("organization does not exist")
        return self.repo.get_or_create_ingestion_source(
            registration.organization_id,
            name=registration.name,
            source_type=registration.source_type,
            external_ref=registration.external_ref,
            config=registration.config,
        )

    def ingest(
        self,
        organization_id: str,
        *,
        source_type: SourceType,
        payload: Any,
        source_name: str,
        **adapter_options: Any,
    ) -> IngestionReceipt:
        adapter = ADAPTERS[source_type]
        batch = adapter.parse(payload, source_name=source_name, **adapter_options)
        return self.ingest_batch(organization_id, batch)

    def ingest_batch(
        self,
        organization_id: str,
        adapter_batch: AdapterBatch,
        *,
        replay_of_batch_id: str | None = None,
    ) -> IngestionReceipt:
        source = self.repo.get_or_create_ingestion_source(
            organization_id,
            name=adapter_batch.source_name,
            source_type=adapter_batch.source_type,
            external_ref=adapter_batch.provenance.get("endpoint") or adapter_batch.provenance.get("topic"),
            config=None,
        )
        batch = self.repo.create_ingestion_batch(
            organization_id,
            source_id=source.id,
            source_type=adapter_batch.source_type,
            idempotency_key=adapter_batch.batch_idempotency_key,
            source_uri=adapter_batch.source_uri,
            provenance=adapter_batch.provenance,
            replay_of_batch_id=replay_of_batch_id,
        )
        failures: list[IngestionFailureReceipt] = []

        for record in adapter_batch.records:
            record_key = self._record_key(source.id, record)
            if self.repo.get_ingested_record_by_key(organization_id, source_id=source.id, idempotency_key=record_key):
                batch.duplicate_count += 1
                continue
            try:
                sensor = self._resolve_sensor(organization_id, record)
                observed_at = self._normalize_timestamp(record.observed_at, record.source_timezone)
                if record.kind == "scalar":
                    target = self._persist_scalar(organization_id, source.name, sensor, observed_at, record)
                    target_type = "scalar_reading"
                    batch.scalar_count += 1
                else:
                    target = self._persist_waveform(
                        organization_id,
                        batch.id,
                        source.name,
                        sensor,
                        observed_at,
                        record_key,
                        record,
                    )
                    target_type = "waveform"
                    batch.waveform_count += 1
                self.repo.create_ingested_record(
                    organization_id,
                    source_id=source.id,
                    batch_id=batch.id,
                    idempotency_key=record_key,
                    target_type=target_type,
                    target_id=target.id,
                    observed_at=observed_at,
                    metric=record.metric,
                    quality=record.quality,
                    provenance=self._record_provenance(adapter_batch, record),
                )
                batch.accepted_count += 1
            except Exception as exc:
                quality = "missing" if isinstance(exc, SensorResolutionError) else "bad"
                self.repo.create_ingestion_failure(
                    organization_id,
                    source_id=source.id,
                    batch_id=batch.id,
                    source_record_id=record.source_record_id,
                    quality=quality,
                    reason=str(exc),
                    detail={"exception": exc.__class__.__name__, "kind": record.kind},
                    payload=self._safe_record_payload(record),
                )
                failures.append(
                    IngestionFailureReceipt(
                        source_record_id=record.source_record_id,
                        reason=str(exc),
                        quality=quality,
                    )
                )
                batch.failed_count += 1

        batch.finished_at = datetime.now(UTC)
        if batch.failed_count and batch.accepted_count:
            batch.status = "partial"
        elif batch.failed_count and not batch.accepted_count:
            batch.status = "failed"
        else:
            batch.status = "accepted"
        self.repo.session.flush()
        return IngestionReceipt(
            batch_id=batch.id,
            source_id=source.id,
            status=batch.status,
            accepted_count=batch.accepted_count,
            duplicate_count=batch.duplicate_count,
            failed_count=batch.failed_count,
            scalar_count=batch.scalar_count,
            waveform_count=batch.waveform_count,
            failures=failures,
        )

    def replay_batch(self, organization_id: str, batch_id: str, *, source_name: str = "Replay") -> IngestionReceipt:
        records = []
        for ingested in self.repo.list_ingested_records_for_batch(organization_id, batch_id):
            if ingested.target_type == "scalar_reading":
                target = self.repo.get_machine_reading(organization_id, ingested.target_id)
                if target is None:
                    continue
                records.append(self._scalar_to_record(target, ingested.id))
            elif ingested.target_type == "waveform":
                target = self.repo.get_waveform_record(organization_id, ingested.target_id)
                if target is None:
                    continue
                records.append(self._waveform_to_record(target, ingested.id))
        adapter_batch = ADAPTERS["replay"].parse(records, source_name=source_name, replay_of_batch_id=batch_id)
        return self.ingest_batch(organization_id, adapter_batch, replay_of_batch_id=batch_id)

    def health(self, organization_id: str) -> dict:
        return self.repo.ingestion_health(organization_id)

    def _resolve_sensor(self, organization_id: str, record: CanonicalIngestionRecord) -> Sensor:
        ref = record.sensor
        if ref.sensor_id:
            sensor = self.repo.get_sensor_by_id(organization_id, ref.sensor_id)
        elif ref.sensor_external_ref:
            sensor = self.repo.get_sensor_by_external_ref(organization_id, ref.sensor_external_ref)
        else:
            sensor = self.repo.get_sensor_by_path(
                organization_id,
                site_slug=ref.site_slug or "",
                asset_slug=ref.asset_slug or "",
                component_slug=ref.component_slug or "",
                sensor_slug=ref.sensor_slug or "",
            )
        if sensor is None:
            raise SensorResolutionError("sensor could not be resolved inside this organization")
        return sensor

    def _normalize_timestamp(self, value: datetime | str, source_timezone: str | None) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if value.tzinfo is None:
            try:
                tz = ZoneInfo(source_timezone or "UTC")
            except ZoneInfoNotFoundError as exc:
                raise IngestionError(f"unknown source timezone {source_timezone!r}") from exc
            value = value.replace(tzinfo=tz)
        return value.astimezone(UTC)

    def _persist_scalar(
        self,
        organization_id: str,
        source_name: str,
        sensor: Sensor,
        observed_at: datetime,
        record: CanonicalIngestionRecord,
    ) -> MachineReading:
        assert record.value is not None
        normalized_value, normalized_unit = normalize_unit_value(record.value, record.unit, sensor.unit)
        return self.repo.create_machine_reading(
            organization_id,
            MachineReadingCreate(
                sensor_id=sensor.id,
                observed_at=observed_at,
                metric=record.metric or "value",
                value=normalized_value,
                unit=normalized_unit,
                source=source_name,
                quality=record.quality,
                payload={
                    "source_record_id": record.source_record_id,
                    "ingestion_metadata": record.metadata,
                    "original_unit": record.unit,
                },
            ),
        )

    def _persist_waveform(
        self,
        organization_id: str,
        batch_id: str,
        source_name: str,
        sensor: Sensor,
        observed_at: datetime,
        record_key: str,
        record: CanonicalIngestionRecord,
    ) -> WaveformRecord:
        if record.samples is not None:
            storage_uri, digest, sample_count = self.waveform_store.put_samples(
                organization_id=organization_id,
                batch_id=batch_id,
                record_key=record_key,
                samples=record.samples,
                metadata=record.metadata,
            )
        elif record.storage_uri is not None and record.sample_count is not None:
            storage_uri, digest, sample_count = self.waveform_store.describe_external(
                storage_uri=record.storage_uri,
                sha256=record.sha256,
                sample_count=record.sample_count,
            )
        else:
            raise IngestionError("waveform records require samples or storage_uri plus sample_count")
        return self.repo.create_waveform_record(
            organization_id,
            sensor_id=sensor.id,
            batch_id=batch_id,
            observed_at=observed_at,
            unit=record.unit,
            sampling_rate_hz=record.sampling_rate_hz or sensor.sampling_rate_hz or 0.0,
            sample_count=sample_count,
            source=source_name,
            quality=record.quality,
            storage_uri=storage_uri,
            sha256=digest,
            metadata_json={
                "source_record_id": record.source_record_id,
                "ingestion_metadata": record.metadata,
                "sensor_unit": sensor.unit,
            },
        )

    def _record_key(self, source_id: str, record: CanonicalIngestionRecord) -> str:
        if record.source_record_id:
            return record.source_record_id[:120]
        payload = record.model_dump(mode="json", exclude={"samples"})
        payload["sample_count"] = record.sample_count or (len(record.samples) if record.samples else None)
        raw = json.dumps({"source_id": source_id, "record": payload}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _record_provenance(self, batch: AdapterBatch, record: CanonicalIngestionRecord) -> dict[str, Any]:
        return {
            "source_type": batch.source_type,
            "source_name": batch.source_name,
            "source_uri": batch.source_uri,
            "source_record_id": record.source_record_id,
            "metadata": record.metadata,
        }

    def _safe_record_payload(self, record: CanonicalIngestionRecord) -> dict[str, Any]:
        payload = record.model_dump(mode="json", exclude={"samples"})
        if record.samples is not None:
            payload["samples_omitted"] = True
            payload["sample_count"] = len(record.samples)
        return payload

    def _scalar_to_record(self, target: MachineReading, source_record_id: str) -> CanonicalIngestionRecord:
        return CanonicalIngestionRecord(
            kind="scalar",
            observed_at=target.observed_at,
            sensor={"sensor_id": target.sensor_id},
            source_record_id=f"replay:{source_record_id}",
            metric=target.metric,
            value=target.value,
            unit=target.unit,
            quality=target.quality,
            metadata={"replayed_from": target.id},
        )

    def _waveform_to_record(self, target: WaveformRecord, source_record_id: str) -> CanonicalIngestionRecord:
        return CanonicalIngestionRecord(
            kind="waveform",
            observed_at=target.observed_at,
            sensor={"sensor_id": target.sensor_id},
            source_record_id=f"replay:{source_record_id}",
            unit=target.unit,
            quality=target.quality,
            sampling_rate_hz=target.sampling_rate_hz,
            sample_count=target.sample_count,
            storage_uri=target.storage_uri,
            sha256=target.sha256,
            metadata={"replayed_from": target.id},
        )
