"""Native protocol connectors that feed source payloads into ingestion adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from industrial_ingestion.contracts import IngestionReceipt
from industrial_ingestion.service import IngestionService


class MqttSubscriptionHandle:
    def __init__(self, client: Any):
        self.client = client

    def stop(self) -> None:
        if hasattr(self.client, "loop_stop"):
            self.client.loop_stop()
        if hasattr(self.client, "disconnect"):
            self.client.disconnect()


class MqttIngestionConnector:
    def __init__(self, service: IngestionService, client_factory: Callable[[], Any] | None = None):
        self.service = service
        self.client_factory = client_factory

    def start_subscription(
        self,
        organization_id: str,
        *,
        broker_host: str,
        topics: Iterable[str],
        source_name: str = "MQTT Connector",
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        keepalive: int = 60,
    ) -> MqttSubscriptionHandle:
        client = self._client()
        if username and hasattr(client, "username_pw_set"):
            client.username_pw_set(username, password)

        def on_message(_client, _userdata, message) -> None:
            self.service.ingest(
                organization_id,
                source_type="mqtt",
                payload=message.payload,
                source_name=source_name,
                topic=message.topic,
            )

        client.on_message = on_message
        client.connect(broker_host, port, keepalive)
        for topic in topics:
            client.subscribe(topic)
        client.loop_start()
        return MqttSubscriptionHandle(client)

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        from paho.mqtt.client import CallbackAPIVersion, Client

        return Client(callback_api_version=CallbackAPIVersion.VERSION2)


class OpcUaSubscriptionHandler:
    def __init__(self, service: IngestionService, organization_id: str, source_name: str):
        self.service = service
        self.organization_id = organization_id
        self.source_name = source_name

    def datachange_notification(self, node, value, data) -> None:
        timestamp = getattr(getattr(data, "monitored_item", None), "Value", None)
        timestamp = getattr(timestamp, "SourceTimestamp", None) or getattr(timestamp, "ServerTimestamp", None)
        timestamp = timestamp or datetime.now(UTC)
        self.service.ingest(
            self.organization_id,
            source_type="opcua",
            source_name=self.source_name,
            payload={
                "nodes": [
                    {
                        "node_id": str(node),
                        "server_timestamp": timestamp.isoformat() if timestamp else None,
                        "browse_name": str(node),
                        "value": value,
                        "unit": "g",
                    }
                ]
            },
        )


class OpcUaIngestionConnector:
    def __init__(self, service: IngestionService, client_factory: Callable[[str], Any] | None = None):
        self.service = service
        self.client_factory = client_factory

    async def subscribe_data_changes(
        self,
        organization_id: str,
        *,
        endpoint: str,
        node_ids: Iterable[str],
        source_name: str = "OPC-UA Connector",
        publishing_interval_ms: int = 1000,
    ) -> Any:
        client = self._client(endpoint)
        await client.connect()
        handler = OpcUaSubscriptionHandler(self.service, organization_id, source_name)
        subscription = await client.create_subscription(publishing_interval_ms, handler)
        nodes = [client.get_node(node_id) for node_id in node_ids]
        await subscription.subscribe_data_change(nodes)
        return subscription

    async def poll_once(
        self,
        organization_id: str,
        *,
        endpoint: str,
        node_ids: Iterable[str],
        source_name: str = "OPC-UA Connector",
        unit: str = "g",
    ) -> list[IngestionReceipt]:
        client = self._client(endpoint)
        await client.connect()
        receipts = []
        try:
            for node_id in node_ids:
                node = client.get_node(node_id)
                value = await node.read_value()
                receipts.append(
                    self.service.ingest(
                        organization_id,
                        source_type="opcua",
                        source_name=source_name,
                        payload={
                            "nodes": [
                                {
                                    "node_id": node_id,
                                    "server_timestamp": datetime.now(UTC).isoformat(),
                                    "browse_name": node_id,
                                    "value": value,
                                    "unit": unit,
                                }
                            ],
                            "endpoint": endpoint,
                        },
                    )
                )
        finally:
            await client.disconnect()
        return receipts

    def _client(self, endpoint: str) -> Any:
        if self.client_factory is not None:
            return self.client_factory(endpoint)
        from asyncua import Client

        return Client(endpoint)


class AbbApiConnector:
    def __init__(self, service: IngestionService, client_factory: Callable[..., Any] | None = None):
        self.service = service
        self.client_factory = client_factory

    def fetch_measurements(
        self,
        organization_id: str,
        *,
        base_url: str,
        token: str,
        asset_id: str | None = None,
        source_name: str = "ABB API Connector",
    ) -> IngestionReceipt:
        client = self._client(base_url=base_url, token=token)
        response = client.get(
            "/measurements",
            params={"asset_id": asset_id} if asset_id else None,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        payload = response.json()
        return self.service.ingest(
            organization_id,
            source_type="abb",
            source_name=source_name,
            payload=payload,
            source_uri=f"{base_url.rstrip('/')}/measurements",
        )

    def _client(self, *, base_url: str, token: str) -> Any:
        if self.client_factory is not None:
            return self.client_factory(base_url=base_url, token=token)
        import httpx

        return httpx.Client(base_url=base_url, timeout=30.0)


def run_async(coro):
    return asyncio.run(coro)
