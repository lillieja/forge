import serial
import time

# Identify your port. On Jetson, it's usually /dev/ttyACM0 or /dev/ttyUSB0
ser = serial.Serial('/dev/ttyCH341USB0', 9600, timeout=1)
time.sleep(2) # Wait for Arduino to reset after connection

def set_servo_angle(angle):
    command = f"{angle}\n"
    ser.write(command.encode())
    print(f"Sent: {angle}")

# Example: Sweep the servo
try:
    while True:
        set_servo_angle(0)
        time.sleep(1)
        set_servo_angle(90)
        time.sleep(1)
        set_servo_angle(180)
        time.sleep(1)
except KeyboardInterrupt:
    ser.close()
