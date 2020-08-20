import cv2
import numpy as np


final = []
def lines(image1):
    gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    canimg = cv2.Canny(gray, 100, 200)
    
    lines= cv2.HoughLines(canimg, 1, np.pi/180.0, 400, np.array([]))
    
    
    
    
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(image1,(x1,y1),(x2,y2),(0,0,255),1)
    
    lines = lines[:,0,0]
    lines = np.sort(lines)[::-1]
    final = np.zeros((len(lines)))
    for i in range(len(lines)):
        
        if i != len(lines)-1:
            if i % 2 == 0:
            
                final[i] = (lines[i] + lines[i+1])//2
    
    final = np.delete(final, np.argwhere(final == 0))
    
    return final
