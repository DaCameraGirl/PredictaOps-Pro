"""Centralized role and service-principal permission policy."""

from __future__ import annotations

from typing import Final

PLATFORM_READ: Final = "platform.read"
PLATFORM_MANAGE: Final = "platform.manage"
INGESTION_WRITE: Final = "ingestion.write"
INGESTION_MANAGE: Final = "ingestion.manage"
ANALYTICS_READ: Final = "analytics.read"
ANALYTICS_RUN: Final = "analytics.run"
ML_READ: Final = "ml.read"
ML_EXPERIMENT_RUN: Final = "ml.experiment.run"
ML_MODEL_REGISTER: Final = "ml.model.register"
ML_MODEL_PROMOTE_VALIDATED: Final = "ml.model.promote_validated"
ML_MODEL_PROMOTE_PRODUCTION: Final = "ml.model.promote_production"
SERVING_BIND: Final = "serving.bind"
SERVING_PREDICT: Final = "serving.predict"
PREDICTION_READ: Final = "prediction.read"
MAINTENANCE_READ: Final = "maintenance.read"
MAINTENANCE_MANAGE: Final = "maintenance.manage"
MAINTENANCE_WORK_ORDER_APPROVE: Final = "maintenance.work_order.approve"
MAINTENANCE_CMMS_SYNC: Final = "maintenance.cmms.sync"
SECURITY_MANAGE: Final = "security.manage"
MEMBERS_MANAGE: Final = "members.manage"
OWNERS_MANAGE: Final = "owners.manage"
SECRETS_MANAGE: Final = "secrets.manage"
AUDIT_READ: Final = "audit.read"

VIEWER_PERMISSIONS: Final = {
    PLATFORM_READ,
    ANALYTICS_READ,
    ML_READ,
    PREDICTION_READ,
    MAINTENANCE_READ,
}

TECHNICIAN_PERMISSIONS: Final = VIEWER_PERMISSIONS | {
    MAINTENANCE_MANAGE,
}

ENGINEER_PERMISSIONS: Final = VIEWER_PERMISSIONS | {
    INGESTION_WRITE,
    INGESTION_MANAGE,
    ANALYTICS_RUN,
    ML_EXPERIMENT_RUN,
    ML_MODEL_REGISTER,
    ML_MODEL_PROMOTE_VALIDATED,
    SERVING_PREDICT,
    MAINTENANCE_MANAGE,
    MAINTENANCE_WORK_ORDER_APPROVE,
    MAINTENANCE_CMMS_SYNC,
}

ADMIN_PERMISSIONS: Final = ENGINEER_PERMISSIONS | {
    PLATFORM_MANAGE,
    ML_MODEL_PROMOTE_PRODUCTION,
    SERVING_BIND,
    SECURITY_MANAGE,
    MEMBERS_MANAGE,
    SECRETS_MANAGE,
    AUDIT_READ,
}

OWNER_PERMISSIONS: Final = ADMIN_PERMISSIONS | {OWNERS_MANAGE}

ROLE_PERMISSIONS: Final = {
    "viewer": VIEWER_PERMISSIONS,
    "technician": TECHNICIAN_PERMISSIONS,
    "engineer": ENGINEER_PERMISSIONS,
    "admin": ADMIN_PERMISSIONS,
    "owner": OWNER_PERMISSIONS,
}

SERVICE_ALLOWED_PERMISSIONS: Final = {
    INGESTION_WRITE,
    ANALYTICS_RUN,
    SERVING_PREDICT,
}

HUMAN_ONLY_PERMISSIONS: Final = {
    MAINTENANCE_MANAGE,
    MAINTENANCE_WORK_ORDER_APPROVE,
    MAINTENANCE_CMMS_SYNC,
    ML_MODEL_PROMOTE_PRODUCTION,
    SECURITY_MANAGE,
    MEMBERS_MANAGE,
    OWNERS_MANAGE,
    SECRETS_MANAGE,
    AUDIT_READ,
}


def permissions_for_role(role: str) -> set[str]:
    return set(ROLE_PERMISSIONS.get(role, set()))


def validate_service_permissions(permissions: list[str]) -> list[str]:
    unknown = sorted(set(permissions) - SERVICE_ALLOWED_PERMISSIONS)
    if unknown:
        raise ValueError(f"service principal permissions are not allowed: {', '.join(unknown)}")
    return sorted(set(permissions))

