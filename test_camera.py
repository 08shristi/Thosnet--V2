import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working ❌")
    exit()

print("Camera working ✅ Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()