import cv2
import recognize
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from time import sleep

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image,scaleFactor=1.1, minNeighbors=4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x,y), (w,h),(0, 0, 255), 2)
            face_cropped = frame[y:y + h, x:x + w]

            # Calculate the ratio of the eye region to the face region
            eye_region_height = int(h * 0.2)
            eye_region_width = int(w * 0.4)
            eye_region_x = int(w * 0.3)
            eye_region_y = int(h * 0.4)
            eye_region = face_cropped[eye_region_y:eye_region_y + eye_region_height,
                         eye_region_x:eye_region_x + eye_region_width]

            # Calculate the average intensity of the pixels in the eye region
            eye_intensity = cv2.mean(eye_region)[0]

            # Check if the average intensity is above a threshold
            if eye_intensity > 100:
                cv2.putText(frame,"fake",(20,80),cv2.FONT_HERSHEY_SIMPLEX, 2 ,(255,0,0),2)

            else:
                cv2.putText(frame, "real", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                recognize.recognize()
                main()

        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()