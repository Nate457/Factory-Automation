import cv2
from gesture import GestureTracker 

def main():
    cap = cv2.VideoCapture(0)
    tracker = GestureTracker()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        processed_frame, hand_data, pose_data = tracker.process_frame(frame)
        cv2.imshow("Live Gesture Tracking", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()