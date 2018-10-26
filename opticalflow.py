import cv2
import numpy as np
cap = cv2.VideoCapture("./testvids/hulahooping.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# flowpath = './flowvids/'

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
# print(hsv)
hsv[...,1] = 255
print(hsv)
out = cv2.VideoWriter('./flowvids/vid1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out1 = cv2.VideoWriter('./flowvids/vid11.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# Mat hsv_channels=[0,0,0]

while(1):
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # flow    =   cv.calcOpticalFlowFarneback(    prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags )


    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow1 = cv2.DualTVL1OpticalFlow(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # optical_flow = cv2.DualTVL1OpticalFlow_create()
    # flow = optical_flow.calc(prvs, next, None)
    # print(flow)
    # cv2.imshow('frame1',flow1)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    # print(hsv[])
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # cv2.split( hsv, hsv_channels );
    # cv2.imshow("HSV to gray", hsv_channels[0]);
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # rgbg = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    # out.write(rgb)

    cv2.imshow('frame2',rgb)
    # cv2.imwrite('opticalfb.png',frame2)
    # cv2.imwrite('opticalhsv.png',rgb)   
    out.write(rgb)
    out1.write(hsv)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()