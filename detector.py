import cv2
import numpy as np
import tensorflow as tf
from gpiozero import LED
import time


# GPIO SETUP (gpiozero)

# Pins use BCM numbering by default with gpiozero
INDICATOR_PIN = 17  # LED for NON-HUMAN (turns ON for class 1)
ERROR_PIN = 27      # LED for ERRORS

indicator_led = LED(INDICATOR_PIN)
error_led = LED(ERROR_PIN)

# Make sure both start OFF
indicator_led.off()
error_led.off()


# LOAD CNN MODEL
try:
    model = tf.keras.models.load_model("model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Model load FAILED:", e)
    error_led.on()
    raise SystemExit


# CAMERA SETUP

# Use 0 if using built-in camera, and 1 if using the Logi
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    error_led.on()
    raise SystemExit

print("Camera initialized.")


# CNN INTERPRETATION
def classify_frame(frame):
    """
    Returns:
        0 = human
        1 = non-human
        2 = error
    """
    try:
        # Change (224, 224) to your model's input size if different
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        # CNN returns class probabilities → take argmax as class index
        class_index = int(np.argmax(prediction))

        return class_index

    except Exception as e:
        print("Prediction error:", e)
        return 2   # error state

# MAIN LOOP

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            error_led.on()
            break

        cv2.imshow("Camera Feed", frame)

        # Run CNN on current frame
        result = classify_frame(frame)
        print("Model Output:", result)

        # LED behavior based on CNN output
        if result == 0:
            # HUMAN detected → indicator OFF, no error
            indicator_led.off()
            error_led.off()

        elif result == 1:
            # NON-HUMAN detected → indicator ON, no error
            indicator_led.on()
            error_led.off()

        else:
            # ERROR → error LED ON, indicator OFF
            error_led.on()
            indicator_led.off()

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # CLEANUP
    cap.release()
    cv2.destroyAllWindows()

    # Turn off LEDs on exit
    indicator_led.off()
    error_led.off()

    print("Program ended cleanly.")
