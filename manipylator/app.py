# app.py
import json
from math import pi
from typing import Optional

import paho.mqtt.client as mqtt
import panel as pn
import param


class Q(param.Parameterized):
    """
    Parameterized class representing the 6 joint angles (q1-q6) of a robot arm.
    Each parameter is a float in radians, bounded between -pi and pi.
    """

    q1 = param.Number(default=0, bounds=(-pi, pi))
    q2 = param.Number(default=0, bounds=(-pi, pi))
    q3 = param.Number(default=0, bounds=(-pi, pi))
    q4 = param.Number(default=0, bounds=(-pi, pi))
    q5 = param.Number(default=0, bounds=(-pi, pi))
    q6 = param.Number(default=0, bounds=(-pi, pi))


class Q_UI:
    """
    Panel UI for displaying the joint values of a Q instance in a 2x3 grid.
    Updates automatically when the Q instance changes.
    """

    def __init__(self, qparams: Q, title: Optional[str] = None):
        """
        Initialize the UI for a Q instance.
        Args:
            qparams: The Q instance to display.
            title: Optional title for the panel.
        """
        self.qparams = qparams
        self.title = title
        self.wdgts = [
            pn.indicators.Number(
                name=f"Joint {i + 1}",
                value=0,
                format="{value:.3f} π",
                font_size="min(6vw, 2em)",
                sizing_mode="stretch_both",
            )
            for i in range(6)
        ]

        self.gui = pn.GridBox(*self.wdgts, ncols=3, sizing_mode="stretch_both")
        self._panel = pn.Column(
            pn.pane.Markdown(f"### {self.title}" if self.title else ""),
            self.gui,
            sizing_mode="stretch_width",
        )
        self.qparams.param.watch(self._update_widgets, [f"q{i + 1}" for i in range(6)])
        self._update_widgets(None)

    def _update_widgets(self, event) -> None:
        """
        Update the indicator widgets to reflect the current values in qparams.
        """
        for i, wdgt in enumerate(self.wdgts):
            wdgt.value = getattr(self.qparams, f"q{i + 1}")

    def panel(self) -> pn.Column:
        """
        Return the Panel layout for this Q_UI.
        """
        return self._panel


class StateViewer(param.Parameterized):
    """
    Main application UI for the robot arm controller.
    Holds current and target pose, stepper status, and manages MQTT connection.
    """

    stepper_energized = param.Boolean(default=False)
    pose: Q
    target_pose: Q
    mq: Optional[mqtt.Client]

    def __init__(
        self,
        pose: Optional[Q] = None,
        target_pose: Optional[Q] = None,
        mq: Optional["mqtt.Client"] = None,
        **params,
    ):
        """
        Initialize the StateViewer.
        Args:
            pose: Optional Q instance for current pose.
            target_pose: Optional Q instance for target pose.
            mq: Optional external MQTT client. If not provided, a new one is created.
        """

        super().__init__(**params)
        self.pose = pose if pose is not None else Q()
        self.target_pose = target_pose if target_pose is not None else Q()
        self.pose_panel = Q_UI(self.pose, title="Current Pose")
        self.target_panel = Q_UI(self.target_pose, title="Target Pose")
        if mq is not None:
            self.mq = mq
            self.mq.on_connect = self.on_connect
            self.mq.on_message = self.on_message
            # Assume the user will connect and loop_start externally if they provide mq
        else:
            self._init_mqtt()

    def _init_mqtt(self) -> None:
        """
        Initialize and connect the internal MQTT client.
        """
        self.mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore
        self.mq.on_connect = self.on_connect
        self.mq.on_message = self.on_message
        try:
            self.mq.connect("localhost", 1883, 60)
            self.mq.loop_start()
            print("MQTT connected successfully")
        except Exception as e:
            print(f"Failed to connect to MQTT: {e}")

    def on_connect(
        self, client: mqtt.Client, userdata, flags, reason_code, properties=None
    ) -> None:
        """
        MQTT callback for successful connection. Subscribes to relevant topics.
        """
        print(f"Connected with result code {reason_code}")
        client.subscribe("manipylator/state")
        client.subscribe("manipylator/target")
        client.subscribe("manipylator/stepper_energized")

    def on_message(self, client: mqtt.Client, userdata, msg) -> None:
        """
        MQTT callback for incoming messages. Updates state based on topic.
        """
        try:
            if msg.topic == "manipylator/state":
                d = json.loads(msg.payload)
                self.pose.q1 = d["q1"]
                self.pose.q2 = d["q2"]
                self.pose.q3 = d["q3"]
                self.pose.q4 = d["q4"]
                self.pose.q5 = d["q5"]
                self.pose.q6 = d["q6"]
            elif msg.topic == "manipylator/target":
                d = json.loads(msg.payload)
                self.target_pose.q1 = d["q1"]
                self.target_pose.q2 = d["q2"]
                self.target_pose.q3 = d["q3"]
                self.target_pose.q4 = d["q4"]
                self.target_pose.q5 = d["q5"]
                self.target_pose.q6 = d["q6"]
            elif msg.topic == "manipylator/stepper_energized":
                self.stepper_energized = json.loads(msg.payload)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing message: {e}")

    def view(self) -> pn.Column:
        """
        Return the main Panel layout for the app, including pose, target, and stepper status.
        """
        stepper_status = (
            "Energized" if self.stepper_energized is True else "De-energized"
        )
        return pn.Column(
            self.pose_panel.panel(),
            self.target_panel.panel(),
            pn.pane.Markdown(f"### Stepper Status: {stepper_status}"),
            sizing_mode="stretch_width",
        )


s = StateViewer()


# Create the Panel template
template = pn.template.BootstrapTemplate(title="Robot Arm Control Panel")
template.main.append(s.view)  # type: ignore
template.servable()
