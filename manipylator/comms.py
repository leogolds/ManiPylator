import json
import time
from datetime import datetime, timezone
from time import sleep
from typing import Optional, List, Dict, Callable

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from schemas import (
    DeviceAboutV1,
    DeviceStatusV1,
    DeviceType,
    DeviceCapability,
    StateStr,
)


class MQTTConnection:
    def __init__(self, host="localhost"):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # self.client.connect("localhost", 1883, 60)
        # self.client.connect("192.168.1.8", 1883, 60)
        # self.client.connect("192.168.1.48", 1883, 60)
        self.client.connect(host, 1883, 60)

        self.client.loop_start()

    def run_gcode_script(self, script: str):
        api_topic = "manipylator/moonraker/api/request"
        payload = {
            "jsonrpc": "2.0",
            "method": "printer.gcode.script",
            "params": {"script": script},
        }

        self.client.publish(
            api_topic,
            json.dumps(payload),
        )

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("manipylator/klipper/alert")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))


class MQClient:
    """Base class for MQTT demo clients with common functionality."""

    def __init__(
        self,
        client_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        subscriptions: List[str] = None,
        message_handlers: Dict[str, Callable] = None,
        device_id: str = None,
        device_type: DeviceType = DeviceType.other,
        device_vendor: str = "Demo Corp",
        device_model: str = "DemoDevice-1000",
        device_capabilities: List[DeviceCapability] = None,
        device_endpoints: Dict[str, str] = None,
        device_owner: str = "demo_user",
    ):
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.running = False
        self.subscriptions = subscriptions or []
        self.message_handlers = message_handlers or {}

        # Device information
        self.device_id = device_id or client_id
        self.device_type = device_type
        self.device_vendor = device_vendor
        self.device_model = device_model
        self.device_capabilities = device_capabilities or []
        self.device_endpoints = device_endpoints or {}
        self.device_owner = device_owner
        self.start_time = time.time()

    def log(self, message: str):
        """Log a message with consistent timestamp formatting."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[
            :-3
        ]  # HH:MM:SS.milliseconds
        print(f"[{timestamp}] [{self.client_id}] {message}")

    def on_connect(self, client, userdata, flags, rc):
        self.log(f"Connected to MQTT broker with result code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.log(f"Disconnected from MQTT broker with result code {rc}")

    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            # Parse the message
            payload = json.loads(msg.payload.decode())
            message_schema = payload.get("message_schema")

            # Call registered handler or default handler
            if message_schema in self.message_handlers:
                self.message_handlers[message_schema](payload)
            else:
                self.handle_message(payload, message_schema)

        except (json.JSONDecodeError, ValidationError) as e:
            self.log(f"Error parsing message: {e}")
        except Exception as e:
            self.log(f"Error processing message: {e}")

    def handle_message(self, payload: dict, message_schema: str):
        """Handle specific message types. Override in subclasses."""
        self.log(f"Received message with schema: {message_schema}")

    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            self.log(f"Failed to connect: {e}")

    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload, retain: bool = False):
        """Publish message to MQTT topic. Accepts Pydantic models or dicts."""
        try:
            # Handle Pydantic models by converting to dict
            if hasattr(payload, "json_serializable_dict"):
                payload = payload.json_serializable_dict()
            elif hasattr(payload, "dict"):
                payload = payload.dict()
            elif not isinstance(payload, dict):
                # Try to convert to dict if it's not already
                payload = dict(payload)

            message = json.dumps(payload)
            result = self.client.publish(topic, message, retain=retain)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                self.log(f"Failed to publish to {topic}: {result.rc}")
        except Exception as e:
            self.log(f"Error publishing to {topic}: {e}")

    def publish_device_about(self):
        """Publish device discovery information."""
        device_about = DeviceAboutV1(
            device_id=self.device_id,
            type=self.device_type,
            vendor=self.device_vendor,
            model=self.device_model,
            capabilities=self.device_capabilities,
            endpoints=self.device_endpoints,
            owner=self.device_owner,
        )

        self.publish(device_about.topic, device_about, retain=True)
        self.log(f"Published device about to {device_about.topic}")

    def publish_device_status(self, state: StateStr = StateStr.online):
        """Publish device status."""
        uptime_seconds = (
            int(time.time() - self.start_time) if state == StateStr.online else 0
        )

        device_status = DeviceStatusV1(
            device_id=self.device_id,
            state=state,
            uptime_seconds=uptime_seconds,
        )

        self.publish(device_status.topic, device_status)
        self.log(f"Published device status ({state}) to {device_status.topic}")

    def start(self):
        """Start the client with subscriptions and custom initialization."""
        self.log(f"Starting {self.__class__.__name__}...")
        self.running = True

        # Subscribe to configured topics
        for topic in self.subscriptions:
            self.client.subscribe(topic)
            self.log(f"Subscribed to {topic}")

        # Publish device information
        self.publish_device_about()
        self.publish_device_status()

    def stop(self):
        """Stop the client."""
        self.log(f"Stopping {self.__class__.__name__}...")
        self.running = False
        # Publish offline status
        self.publish_device_status(StateStr.offline)

    def run(self):
        """Run the client with main loop."""
        self.start()

        try:
            # Keep running to process messages
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    conn = MQTTConnection()

    for i in range(10):
        conn.run_gcode_script(f"echo_numbers VALUE={1.5 * i}")
        sleep(1)
