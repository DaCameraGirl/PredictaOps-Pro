# Production Roadmap

The target is the complete production Predictive Maintenance Studio, not a narrow
demo or a "data foundation only" project. The platform should support the full
industrial workflow from machine data ingestion through evidence-backed
maintenance action, with abstention as a first-class model behavior.

## Product Target

The complete system includes:

- Multi-company, multi-site, and multi-asset support
- Asset, component, and sensor registries
- Canonical machine-data schemas
- CSV and Parquet ingestion
- REST, MQTT, and OPC-UA ingestion
- ABB sensor/API integration points
- Raw waveform storage and time-series feature storage
- Real-time feature extraction
- Vibration analysis and FFT
- Degradation and anomaly detection
- RUL models across multiple failure modes
- Model registry, training runs, and experiment tracking
- Leave-one-asset and run-out validation
- Calibrated uncertainty and model abstention
- Model promotion and rollback
- Drift detection and retraining triggers
- Fleet, site, machine, component, and sensor dashboards
- Alerting and maintenance recommendations
- Technician acknowledgement, comments, and history
- Work-order workflow and CMMS integration contracts
- User accounts, organizations, RBAC, SSO/OIDC, audit logs, secrets, and tenant isolation
- Database migrations, background jobs, monitoring, backups, health checks, Docker deployment, and CI/CD
- API documentation, exports, production tests, and browser E2E coverage
- Agentic maintenance copilot with tool use, evidence citations, permissions, and human approval gates

## Flagship Principle: Abstention

A production predictive model must be allowed to say:

> Unsupported / insufficient evidence.

That is not unfinished software. It is the expected behavior when the machinery,
operating conditions, sensor configuration, failure mode, or evidence window is
outside the model's validated domain.

The system should still report what it knows: observed sensor evidence,
degradation state, applicable validation domain, uncertainty limits, missing
evidence, and recommended human review path. It should not convert an
unsupported diagnostic model output into a supported maintenance prediction.

## Consecutive Production Slices

These slices are ordered so every review branch can be reviewed and tested as a
coherent production increment. Slice numbers are product architecture labels;
GitHub pull request numbers may drift after branch renames or replacement PRs.

1. **Production Slice 6 - Platform Core**
   Database, organizations, users, sites, assets, components, sensors, canonical schemas, migrations, and persistence.
2. **Production Slice 7 - Industrial Ingestion**
   CSV, Parquet, REST, MQTT, OPC-UA, ABB/vendor adapters, replay, validation, unit and timezone normalization,
   sensor resolution, ingestion provenance, idempotency, quality states, dead-letter failures, tenant-boundary tests,
   source health, and first-class waveform ingestion metadata.
3. **Production Slice 8 - Analytics Pipeline**
   Streaming feature extraction, waveform processing/storage hardening, FFT, anomaly/degradation engine, and health state.
4. **Production Slice 9 - ML Platform**
   Dataset versions, experiments, model registry, cross-run validation, uncertainty, abstention, promotion, and rollback.
5. **Production Slice 10 - Production Serving**
   Per-asset model resolution, live predictions, drift, monitoring, and retraining triggers.
6. **Production Slice 11 - Maintenance Operations**
   Alerts, cases, inspections, acknowledgement, technician notes, work orders, and CMMS adapter contract.
7. **Production Slice 12 - Enterprise Security**
   RBAC, SSO/OIDC, tenant isolation, audit logs, secrets, and security hardening.
8. **Production Slice 13 - Full Studio UI**
   Fleet to site to machine to bearing to sensor drill-down, health, risk, work-order, and model views.
9. **Production Slice 14 - Agentic Copilot**
   Tool-using maintenance agent, citations, permissions, and human approval gates.
10. **Production Slice 15 - Production Hardening**
    Load tests, failure recovery, backup/restore, observability, docs, deployment, and E2E gauntlet.

The point is not to defer the real system. The point is to build the full system
through reviewable slices so a wrong prediction can be traced to a specific
layer instead of being hidden inside one untestable mega-change.
