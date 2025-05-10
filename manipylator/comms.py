import paho.mqtt.client as mqtt
import json
from time import sleep
from datetime import datetime


class MQTTConnection:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # self.client.connect("localhost", 1883, 60)
        self.client.connect("192.168.1.8", 1883, 60)
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


if __name__ == "__main__":
    conn = MQTTConnection()

    for i in range(10):
        conn.run_gcode_script(f"echo_numbers VALUE={1.5*i}")
        sleep(1)
