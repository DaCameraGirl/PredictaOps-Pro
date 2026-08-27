"""Analytics pipeline service over canonical Platform Core records."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from analytics_pipeline.baseline import choose_health_state, score_feature
from analytics_pipeline.contracts import AnalyticsFailureReceipt, AnalyticsReceipt, FeatureValue
from analytics_pipeline.features import FeatureExtractionError, scalar_features, waveform_features
from analytics_pipeline.waveforms import WaveformIntegrityError, load_waveform
from platform_core.models import AnalyticsFeatureRecord, IngestedRecord, MachineReading, WaveformRecord
from platform_core.repositories import PlatformRepository

ALGORITHM_VERSION = "analytics-v1"


class AnalyticsError(ValueError):
    pass


class AnalyticsService:
    def __init__(self, session: Session):
        self.repo = PlatformRepository(session)

    def compute_batch(self, organization_id: str, batch_id: str) -> AnalyticsReceipt:
        batch = self.repo.get_ingestion_batch(organization_id, batch_id)
        if batch is None:
            raise AnalyticsError("ingestion batch does not exist inside this organization")
        run = self.repo.create_analytics_run(
            organization_id,
            run_kind="batch",
            algorithm_version=ALGORITHM_VERSION,
            input_batch_id=batch_id,
            provenance={"input": "ingestion_batch", "batch_id": batch_id},
        )
        records = self.repo.list_ingested_records_for_batch(organization_id, batch_id)
        receipt = self._compute_records(organization_id, run.id, records)
        self._finish_run(run, receipt)
        return receipt

    def recompute_sensor(self, organization_id: str, sensor_id: str) -> AnalyticsReceipt:
        if self.repo.get_sensor_by_id(organization_id, sensor_id) is None:
            raise AnalyticsError("sensor does not exist inside this organization")
        run = self.repo.create_analytics_run(
            organization_id,
            run_kind="sensor",
            algorithm_version=ALGORITHM_VERSION,
            sensor_id=sensor_id,
            provenance={"input": "sensor_recompute", "sensor_id": sensor_id},
        )
        records = [
            *_records_from_readings(self.repo.list_machine_readings_for_sensor(organization_id, sensor_id)),
            *_records_from_waveforms(self.repo.list_waveform_records_for_sensor(organization_id, sensor_id)),
        ]
        records.sort(key=lambda record: (record.observed_at, record.target_type, record.target_id))
        receipt = self._compute_records(organization_id, run.id, records)
        self._finish_run(run, receipt)
        return receipt

    def health(self, organization_id: str) -> dict:
        states = self.repo.latest_analytics_health(organization_id)
        return {
            "organization_id": organization_id,
            "sensor_count": len(states),
            "states": [
                {
                    "sensor_id": row.sensor_id,
                    "observed_at": row.observed_at.isoformat(),
                    "health_state": row.health_state,
                    "anomaly_score": row.anomaly_score,
                    "trend_slope": row.trend_slope,
                    "confidence": row.confidence,
                    "evidence": row.evidence,
                }
                for row in states
            ],
        }

    def _compute_records(
        self,
        organization_id: str,
        run_id: str,
        records: list[IngestedRecord],
    ) -> AnalyticsReceipt:
        processed_count = 0
        feature_count = 0
        duplicate_feature_count = 0
        health_state_count = 0
        failures: list[AnalyticsFailureReceipt] = []

        for record in records:
            try:
                created = self._process_record(organization_id, run_id, record)
                processed_count += 1
                feature_count += created["features"]
                duplicate_feature_count += created["duplicates"]
                health_state_count += created["health_states"]
            except (AnalyticsError, FeatureExtractionError, WaveformIntegrityError) as exc:
                source_kind = "waveform" if record.target_type == "waveform" else "scalar"
                self.repo.create_analytics_failure(
                    organization_id,
                    run_id=run_id,
                    sensor_id=None,
                    batch_id=record.batch_id,
                    source_kind=source_kind,
                    source_record_id=record.target_id,
                    reason=str(exc),
                    detail={
                        "exception": exc.__class__.__name__,
                        "ingested_record_id": record.id,
                        "target_type": record.target_type,
                    },
                )
                failures.append(
                    AnalyticsFailureReceipt(
                        source_kind=source_kind,
                        source_record_id=record.target_id,
                        reason=str(exc),
                    )
                )

        status = "completed"
        if failures and processed_count:
            status = "partial"
        elif failures and not processed_count:
            status = "failed"
        return AnalyticsReceipt(
            run_id=run_id,
            status=status,
            algorithm_version=ALGORITHM_VERSION,
            processed_count=processed_count,
            feature_count=feature_count,
            duplicate_feature_count=duplicate_feature_count,
            failure_count=len(failures),
            health_state_count=health_state_count,
            failures=failures,
        )

    def _process_record(self, organization_id: str, run_id: str, record: IngestedRecord) -> dict[str, int]:
        if record.target_type == "scalar_reading":
            reading = self.repo.get_machine_reading(organization_id, record.target_id)
            if reading is None:
                raise AnalyticsError("source scalar reading is missing")
            features = scalar_features(reading.metric, reading.value, reading.unit)
            created_features = self._persist_features(
                organization_id,
                run_id,
                source_kind="scalar",
                source=reading,
                batch_id=record.batch_id,
                features=features,
                provenance={"ingested_record_id": record.id, "quality": record.quality},
            )
        elif record.target_type == "waveform":
            waveform = self.repo.get_waveform_record(organization_id, record.target_id)
            if waveform is None:
                raise AnalyticsError("source waveform record is missing")
            loaded = load_waveform(waveform)
            features = waveform_features(loaded.samples, unit=waveform.unit, sampling_rate_hz=waveform.sampling_rate_hz)
            created_features = self._persist_features(
                organization_id,
                run_id,
                source_kind="waveform",
                source=waveform,
                batch_id=record.batch_id,
                features=features,
                provenance={
                    "ingested_record_id": record.id,
                    "quality": record.quality,
                    "content_sha256": loaded.content_sha256,
                    "checksum_verified": loaded.checksum_verified,
                },
            )
        else:
            raise AnalyticsError(f"unsupported ingested target type {record.target_type!r}")

        current_features = [feature for feature, _created in created_features]
        health_states = self._persist_health_state(organization_id, run_id, current_features)
        return {
            "features": sum(1 for _feature, created in created_features if created),
            "duplicates": sum(1 for _feature, created in created_features if not created),
            "health_states": health_states,
        }

    def _persist_features(
        self,
        organization_id: str,
        run_id: str,
        *,
        source_kind: str,
        source: MachineReading | WaveformRecord,
        batch_id: str | None,
        features: list[FeatureValue],
        provenance: dict,
    ) -> list[tuple[AnalyticsFeatureRecord, bool]]:
        persisted: list[tuple[AnalyticsFeatureRecord, bool]] = []
        for feature in features:
            existing = self.repo.get_analytics_feature(
                organization_id,
                algorithm_version=ALGORITHM_VERSION,
                source_kind=source_kind,
                source_record_id=source.id,
                feature_name=feature.name,
            )
            if existing:
                persisted.append((existing, False))
                continue
            row = self.repo.create_analytics_feature(
                organization_id,
                run_id=run_id,
                sensor_id=source.sensor_id,
                batch_id=batch_id,
                source_kind=source_kind,
                source_record_id=source.id,
                observed_at=source.observed_at,
                feature_name=feature.name,
                value=feature.value,
                unit=feature.unit,
                quality=source.quality,
                algorithm_version=ALGORITHM_VERSION,
                provenance=provenance,
            )
            persisted.append((row, True))
        return persisted

    def _persist_health_state(
        self,
        organization_id: str,
        run_id: str,
        current_features: list[AnalyticsFeatureRecord],
    ) -> int:
        if not current_features:
            return 0
        by_sensor_time: dict[tuple[str, datetime], list[AnalyticsFeatureRecord]] = {}
        for feature in current_features:
            by_sensor_time.setdefault((feature.sensor_id, feature.observed_at), []).append(feature)

        created = 0
        for (sensor_id, observed_at), features in by_sensor_time.items():
            scores = []
            for feature in features:
                history = self.repo.list_analytics_features_for_sensor(
                    organization_id,
                    sensor_id,
                    algorithm_version=ALGORITHM_VERSION,
                    feature_name=feature.feature_name,
                )
                scores.append(score_feature(feature, history))
            health_state, anomaly_score, trend_slope, confidence, evidence = choose_health_state(scores)
            self.repo.create_analytics_health_state(
                organization_id,
                run_id=run_id,
                sensor_id=sensor_id,
                observed_at=observed_at,
                health_state=health_state,
                anomaly_score=anomaly_score,
                trend_slope=trend_slope,
                confidence=confidence,
                algorithm_version=ALGORITHM_VERSION,
                evidence=evidence,
            )
            created += 1
        return created

    def _finish_run(self, run, receipt: AnalyticsReceipt) -> None:
        run.finished_at = datetime.now(UTC)
        run.status = receipt.status
        run.feature_count = receipt.feature_count
        run.failure_count = receipt.failure_count
        run.health_state_count = receipt.health_state_count
        self.repo.session.flush()


def _records_from_readings(readings: list[MachineReading]) -> list[IngestedRecord]:
    return [
        IngestedRecord(
            organization_id=row.organization_id,
            source_id="analytics-recompute",
            batch_id=None,
            idempotency_key=f"analytics-recompute:{row.id}",
            target_type="scalar_reading",
            target_id=row.id,
            observed_at=row.observed_at,
            metric=row.metric,
            quality=row.quality,
            provenance={"source": "sensor_recompute"},
        )
        for row in readings
    ]


def _records_from_waveforms(waveforms: list[WaveformRecord]) -> list[IngestedRecord]:
    return [
        IngestedRecord(
            organization_id=row.organization_id,
            source_id="analytics-recompute",
            batch_id=row.batch_id,
            idempotency_key=f"analytics-recompute:{row.id}",
            target_type="waveform",
            target_id=row.id,
            observed_at=row.observed_at,
            metric=None,
            quality=row.quality,
            provenance={"source": "sensor_recompute"},
        )
        for row in waveforms
    ]
