import cv2
import numpy as np


###### OpenCV position of click
point = []
def click_mouse(event,x,y,flags,params):
  # grab references to the global variables
	global point
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		point = [(x, y)]

## TODO
#def roi(x,y,img):
    ## todo write code for draw a rectangle of ones (mask)



path_image='/app/data/GrabCut/data_GT/'
file_name='bool.jpg'
frame=cv2.imread(path_image+file_name)
cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

## click issues
cv2.namedWindow("original")
cv2.setMouseCallback("original", click_mouse)

# Display
while(True):
  cv2.imshow('original',frame)

  if point:
    #print(np.shape(point))
    #print(point[0][0])
    print('pixel position: '+str(point) +' value of mask: '+ str(res[point[0][1],point[0][0]]) )
    ## code for roi

    ##roi_img=roi(x,y,frame)

    #cv2.imshow('frame',res)
    #cv2.imshow('frame2',img)
  # Press Q on keyboard to stop recording
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Closes all the frames
cv2.destroyAllWindows()   
