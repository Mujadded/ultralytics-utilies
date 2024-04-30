import cv2
capture = cv2.VideoCapture('car_parts/car.mp4')
frameNr = 1
while (True):
  success, frame = capture.read()	
  if success:
        cv2.imwrite(f'/Users/mujaddedalif/src/video_split/car_parts/dataset/frame_{frameNr}.jpg', frame)
  else:
      break
 
  frameNr = frameNr+1

print(frameNr+1)
 
capture.release()
  