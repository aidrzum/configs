import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from pyzbar.pyzbar import decode
import subprocess

# Function to execute command
def execute_command(command):
    try:
        print(f"Executing command: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


picam2 = Picamera2()
picam2.start_preview(Preview.NULL)
config = picam2.create_preview_configuration(main={"size": (1280, 960)})
picam2.configure(config)
picam2.start()

# 
# # Initialize the camera
# picam2 = Picamera2()
# #picam2.start_preview(Preview.QT)
# #config = picam2.create_preview_configuration(main={"size": (640, 480)})
# #picam2.configure(config)
# picam2.start()

# Main loop

try:
    while True:
        # Capture an image from the camera
        image = picam2.capture_array()

        # Convert the image to grayscale for QR code detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Decode QR codes in the image
        decoded_objects = decode(gray_image)

        for obj in decoded_objects:
            # Get the data from the QR code
            qr_data = obj.data.decode('utf-8')
            print(f"QR Code detected: {qr_data}")

            # Execute a command based on the QR code data
            execute_command(qr_data)

        # Display the image with detected QR codes (optional)
        for obj in decoded_objects:
            (x, y, w, h) = obj.rect
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, obj.data.decode('utf-8'), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.imshow("QR Code Scanner", image)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()


