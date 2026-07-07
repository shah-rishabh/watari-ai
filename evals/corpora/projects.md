# Projects

## Project Todo: Autonomous Greenhouse
Tidal is Mara's autonomous greenhouse controller. It regulates temperature,
humidity, and irrigation for a small rooftop farm using a network of ESP32
sensors. The control loop runs every 90 seconds. Tidal is written in Rust and
stores readings in a local TimescaleDB instance. The project began in March
2023 and is deployed on Mara's own rooftop.

## Project Kestrel: Warehouse Drone
Kestrel is an indoor inventory drone that scans barcode labels on warehouse
shelves. It navigates using visual-inertial odometry rather than GPS, because it
operates entirely indoors. Kestrel's battery lasts 22 minutes per flight. The
computer vision stack uses a quantized YOLO model running on a Jetson Orin Nano.
Kestrel is a client project for a logistics company and is under NDA.

## Project Loom: Knitting Machine
Loom is a hobby project: a retrofitted industrial knitting machine that Mara
controls from a web interface. It can produce a scarf in about 40 minutes. Loom
uses a Raspberry Pi 4 as its controller and communicates with the motors over
a CAN bus. It is not connected to the internet by design.

## Tooling Preferences
Across projects Mara standardizes on Rust for embedded firmware and Python for
data analysis. She uses Git with a trunk-based workflow and requires that every
firmware change include a hardware-in-the-loop test before merge.
