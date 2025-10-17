import sys, os, time, cv2
sys.path.append("/usr/lib/python3/dist-packages")

from picamera2 import Picamera2
from ai_model import classify_frame

# initialize camera and configure
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (640, 480)})
cam.configure(config)
cam.start()

print("♻️ Smart Waste Classifier – Press SPACE to capture, Q to quit.\n")

while True:
    time.sleep(0.2)
    frame = cam.capture_array()

    # show live preview (if monitor exists)
    if os.environ.get("DISPLAY"):
        cv2.putText(frame, "Press SPACE to classify", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Smart Trash Bin", frame)

    key = cv2.waitKey(1) & 0xFF if os.environ.get("DISPLAY") else ord(input("➡️ Press ENTER to capture...")[:1] or " ")

    if key == ord('q'):
        print("🧠 Exiting.")
        break

    elif key == 32 or key == ord(' '):  # SPACE pressed
        print("📸 Capturing fresh frame...")
        
        # ✅ flush old frames & capture new one
        cam.capture_metadata()  
        frame = cam.capture_array()  # capture fresh frame
        
        label, conf = classify_frame(frame)
        print(f"🧾 Detected: {label} ({conf:.2f})")

        filename = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}\n")
        time.sleep(1)

cam.close()
cv2.destroyAllWindows()
