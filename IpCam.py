import cv2

#i=0
#while(i<2):
    #cap = cv2.VideoCapture('http://192.168.43.1:4747/video')
    #ret,frame = cap.read()
    #print(ret)

# Use the next line if your camera has a username and password
#stream = cv2.VideoCapture("http://192.168.4.1:80/video")
stream = cv2.VideoCapture(0)

while True:
    r, f = stream.read()
    cv2.imshow('IP Camera stream',f)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()