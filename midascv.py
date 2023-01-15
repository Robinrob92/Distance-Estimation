import cv2
import torch
import matplotlib.pyplot as plt

#download the MiDas
midas=torch.hub.load('intel-isl/MidaS', 'NiDas_small')
midas.to('cpu')
midas.eval()

#input transformational pipeline
transforms = torch.hub.load('intel-isl/MidaS', 'transforms')
transform=transforms.small_transform


#hook into OpenCV
cap=cv2.VideoCapture(0)
while cap.isOPened():
    ret, frame=cap.read()
    cv2.imshow('CV2Frame', frame)
    
    if cv2.waitKey(10)&0xFF==ord('q'):
        cap.release()
        cv2.destroyAllWindows()


