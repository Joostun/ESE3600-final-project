#!/usr/bin/env python3
import serial
import argparse
import os
import re
from datetime import datetime
from serial import SerialException

def read_exact(ser, n):
    """Read exactly n bytes from serial."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            # 0 bytes -> usually disconnect / error
            raise RuntimeError("Serial read returned no data before size reached")
        buf.extend(chunk)
    return bytes(buf)

def wait_for_startimg(ser):
    """Wait for a line like 'STARTIMG <size>' and return the size."""
    while True:
        line = ser.readline()
        if not line:
            continue
        decoded = line.decode("utf-8", errors="ignore").strip()
        # Uncomment if you want to see all debug text:
        # print("DEBUG:", decoded)
        if decoded.startswith("STARTIMG"):
            print(f"Got header: {decoded}")
            m = re.match(r"^STARTIMG\s+(\d+)", decoded)
            if not m:
                print("Malformed STARTIMG line, ignoring:", decoded)
                continue
            size = int(m.group(1))
            print(f"Expecting {size} bytes of image data...")
            return size

def main():
    parser = argparse.ArgumentParser(description="Receive JPEG images over serial from ESP32 (looping)")
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port (e.g. COM3, /dev/ttyUSB0, /dev/ttyACM0, /dev/cu.usbmodem*)"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate (must match Serial.begin on ESP32)"
    )
    parser.add_argument(
        "--outdir",
        default="images",
        help="Directory to save received images"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except SerialException as e:
        print(f"Failed to open serial port: {e}")
        return

    print("Connected.")
    print("Each time you want a picture:")
    print("  1) Press Enter in this terminal")
    print("  2) Script sends 'r' to the ESP32")
    print("  3) Image is received and saved\n")

    img_index = 0

    try:
        while True:
            input("Press Enter to request an image (Ctrl+C to quit)...")

            # Send 'r' to trigger capture on the ESP32
            try:
                ser.write(b"r")
            except SerialException as e:
                print(f"Serial write failed: {e}")
                break

            print("\n--- Waiting for STARTIMG header ---")
            try:
                size = wait_for_startimg(ser)
            except RuntimeError as e:
                print(f"Error while waiting for STARTIMG: {e}")
                break
            except SerialException as e:
                print(f"Serial error while waiting for STARTIMG: {e}")
                break

            # Read image data
            try:
                img_bytes = read_exact(ser, size)
            except (RuntimeError, SerialException) as e:
                print(f"Error reading image data: {e}")
                break

            print(f"Read {len(img_bytes)} bytes of image data.")

            # Look for ENDIMG (optional sanity check)
            print("Looking for ENDIMG marker...")
            try:
                while True:
                    line = ser.readline()
                    if not line:
                        # timeout; we won't loop forever here
                        break
                    decoded = line.decode("utf-8", errors="ignore").strip()
                    if decoded == "ENDIMG":
                        print("Got ENDIMG.")
                        break
            except SerialException as e:
                print(f"Serial error while looking for ENDIMG: {e}")
                break

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(args.outdir, f"capture_{timestamp}_{img_index:03d}.jpg")
            with open(filename, "wb") as f:
                f.write(img_bytes)

            print(f"Image #{img_index} saved to: {filename}\n")
            img_index += 1

    except KeyboardInterrupt:
        print("\nExiting on user request (Ctrl+C).")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
