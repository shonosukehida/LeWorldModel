import threading
import time
from collections import deque

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xarm.wrapper import XArmAPI


ROBOT_IP = "192.168.1.240"

# Sensor sampling frequency
SAMPLE_HZ = 20.0

# Number of seconds shown in the graph
WINDOW_SECONDS = 20.0

MAX_SAMPLES = int(SAMPLE_HZ * WINDOW_SECONDS)


# ------------------------------------------------------------
# Shared buffers
# ------------------------------------------------------------

lock = threading.Lock()

timestamps = deque(maxlen=MAX_SAMPLES)

fx_data = deque(maxlen=MAX_SAMPLES)
fy_data = deque(maxlen=MAX_SAMPLES)
fz_data = deque(maxlen=MAX_SAMPLES)

tx_data = deque(maxlen=MAX_SAMPLES)
ty_data = deque(maxlen=MAX_SAMPLES)
tz_data = deque(maxlen=MAX_SAMPLES)


# ------------------------------------------------------------
# xArm / FT sensor
# ------------------------------------------------------------

arm = XArmAPI(
    ROBOT_IP,
    enable_report=True,
)

running = True


def check_code(code: int, name: str) -> None:
    if code != 0:
        raise RuntimeError(
            f"{name} failed with code={code}"
        )


def initialize_ft_sensor() -> None:
    print("Initializing xArm FT sensor...")

    check_code(
        arm.motion_enable(enable=True),
        "motion_enable",
    )

    check_code(
        arm.set_ft_sensor_enable(0),
        "disable FT sensor",
    )

    check_code(
        arm.clean_error(),
        "clean_error",
    )

    check_code(
        arm.clean_warn(),
        "clean_warn",
    )

    check_code(
        arm.set_ft_sensor_enable(1),
        "enable FT sensor",
    )

    time.sleep(0.5)

    code, config = arm.get_ft_sensor_config()
    check_code(
        code,
        "get_ft_sensor_config",
    )

    print("FT sensor enabled.")
    print("FT sensor config:", config)


def sensor_loop() -> None:
    global running

    t0 = time.monotonic()
    interval = 1.0 / SAMPLE_HZ

    while running:
        loop_start = time.monotonic()

        try:
            code, ft = arm.get_ft_sensor_data()

            if code != 0:
                print(
                    f"get_ft_sensor_data failed: "
                    f"code={code}"
                )
                time.sleep(0.1)
                continue

            fx, fy, fz, tx, ty, tz = ft

            elapsed = time.monotonic() - t0

            with lock:
                timestamps.append(elapsed)

                fx_data.append(fx)
                fy_data.append(fy)
                fz_data.append(fz)

                tx_data.append(tx)
                ty_data.append(ty)
                tz_data.append(tz)

        except Exception as exc:
            print(
                f"FT sensor read error: {exc}"
            )

        elapsed_loop = (
            time.monotonic() - loop_start
        )

        sleep_time = interval - elapsed_loop

        if sleep_time > 0:
            time.sleep(sleep_time)


# ------------------------------------------------------------
# Dash
# ------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2(
            "xArm 6-Axis Force / Torque Sensor"
        ),

        html.Div(
            f"Sampling: {SAMPLE_HZ:.0f} Hz "
            f"| Window: {WINDOW_SECONDS:.0f} s"
        ),

        dcc.Graph(
            id="ft-graph",
            style={
                "height": "1300px",
            },
        ),

        dcc.Interval(
            id="update-timer",
            interval=100,
            n_intervals=0,
        ),
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
    },
)


@app.callback(
    Output("ft-graph", "figure"),
    Input("update-timer", "n_intervals"),
)
def update_graph(_):
    with lock:
        t = list(timestamps)

        series = [
            list(fx_data),
            list(fy_data),
            list(fz_data),
            list(tx_data),
            list(ty_data),
            list(tz_data),
        ]

    labels = [
        "Fx [N]",
        "Fy [N]",
        "Fz [N]",
        "Tx [Nm]",
        "Ty [Nm]",
        "Tz [Nm]",
    ]

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=labels,
    )

    for i, (label, values) in enumerate(
        zip(labels, series),
        start=1,
    ):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=values,
                mode="lines",
                name=label,
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        fig.update_yaxes(
            title_text=label,
            row=i,
            col=1,
        )

    fig.update_xaxes(
        title_text="Time [s]",
        row=6,
        col=1,
    )

    if len(t) >= 2:
        fig.update_xaxes(
            range=[
                max(0.0, t[-1] - WINDOW_SECONDS),
                t[-1],
            ]
        )

    fig.update_layout(
        title="Real-time Force / Torque",
        height=1250,
        margin=dict(
            l=80,
            r=30,
            t=80,
            b=60,
        ),
    )

    return fig


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    global running

    try:
        initialize_ft_sensor()

        thread = threading.Thread(
            target=sensor_loop,
            daemon=True,
        )

        thread.start()

        print()
        print("Web visualization started.")
        print("Server port: 8050")
        print()

        app.run(
            host="127.0.0.1",
            port=8050,
            debug=False,
        )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        running = False

        try:
            arm.set_ft_sensor_enable(0)
        except Exception:
            pass

        arm.disconnect()

        print("Disconnected.")


if __name__ == "__main__":
    main()