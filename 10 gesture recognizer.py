import tkinter 
from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import math
import skimage
from skimage.measure import label,regionprops,regionprops_table
from pyefd import elliptic_fourier_descriptors
import time
from sklearn.ensemble import RandomForestClassifier
from skimage.color import label2rgb
from socket import *
import pickle
import sys
import mahotas

try:
 ippc="192.168.43.91"
 iprasp="192.168.43.74"
 addr = (iprasp,15000) #rasp
 UDPSock = socket(AF_INET, SOCK_DGRAM)
 addr1 = (ippc, 12000) #pc
 UDPSock1 = socket(AF_INET, SOCK_DGRAM)
 UDPSock1.bind(addr1)
except :
     pass
off1,r=17,45
img_size=224
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
face_cascade = cv2.CascadeClassifier('C:/Users/djamel eddine/Desktop/py/haarcascade_frontalface_default.xml')
img_hand=cv2.imread("C:/Users/djamel eddine/Desktop/py/digit51.jpg")
clf1 = pickle.load(open('rf102car.sav', 'rb'))
categories=["digit1","digit1p","digit2","digit2p","digit3","digit3p","digit4","digit4p","digit5","pince"]
pi=math.pi
orb = cv2.ORB_create()
img_hand_resiz2=cv2.resize(img_hand,(440,480))
img_handG=cv2.cvtColor(img_hand_resiz2,cv2.COLOR_BGR2GRAY)
_,img_handt=cv2.threshold(img_handG,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU+cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
op = cv2.morphologyEx(img_handt, cv2.MORPH_OPEN,es)
cls = cv2.morphologyEx(op, cv2.MORPH_CLOSE,es)
clsD = cv2.dilate(cls, es, iterations = 1)
mask_init = cv2.GaussianBlur(clsD,(5,5),100)
skin_img = cv2.bitwise_and(img_hand_resiz2,img_hand_resiz2, mask=mask_init)
kpskin_image, desskin_image = orb.detectAndCompute(skin_img , None)
#-------------------init------------------------------------------------------------------------------------
img_hand_resiz1=cv2.resize(img_hand,(150 ,200))
kp1, des1 = orb.detectAndCompute(img_hand_resiz1, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
h,w=530 ,545
c=-1

class choice:
     
     def __init__(self,window, window_title):
      self.window = window
      self.window.title(window_title)
      
      self.style1 = ttk.Style()
      self.style1.theme_use('alt')
    
      self.style1.configure('B1.TButton',
        highlightthickness='20',
        font=('roman', 12, 'bold'))
      self.style1.map("B1.TButton",
       foreground=[('pressed', 'red'),('!disabled','yellow') ],
       background=[('pressed',  'black'), ('!disabled','black')],           
       )
      
      self.btn_choice1=ttk.Button(window, text="pi camera",width=20,command=self.raspberry,style="B1.TButton")
      self.btn_choice1.pack(side=tkinter.LEFT)
      
      self.style2 = ttk.Style()
      self.style2.theme_use('alt')
      self.style2.configure('B2.TButton',
        highlightthickness='20',
        font=('roman', 12, 'bold'))
      self.style2.map("B2.TButton",
       foreground=[('pressed', 'orange'),('!disabled','cyan') ],
       background=[('pressed',  'black'), ('!disabled','black')],           
       )
      self.btn_choice2=ttk.Button(window, text="pc camera", width=20, command=self.pc,style="B2.TButton")
      self.btn_choice2.pack(side=tkinter.LEFT)
      
      self.window.mainloop()
     def raspberry(self):
         global c
         c=0
         self.window.destroy()
     def pc (self):
          global c
          c=1
          self.window.destroy()
class App:
    a=0
    def __init__(self,window, window_title,s,video_source=0):
      self.s=s
      self.window = window
      self.window.title(window_title)
      self.video_source = video_source
      self.vid = MyVideoCapture(0,s,self.video_source)
      self.canvas = tkinter.Canvas(window, width = w, height = h, bg = "black") 
      self.canvas.pack()       
      self.delay = 1
      self.update()
      self.window.mainloop()


         
    def update(self):
         global upper,lower,contourarea,h,w
         #----------------track-------------------------------------------------------------------------------------
        
         if self.s==1:
          try:    
             ret,frame,roiframe,mask,lower,upper = self.vid.get_frame(1)
             
             if ret:
                 try:
                  
                  contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                  cnt = max(contours, key = lambda x: cv2.contourArea(x))
                  epsilon = 0.0005*cv2.arcLength(cnt,True)
                  approx= cv2.approxPolyDP(cnt,epsilon,True)
                  hull = cv2.convexHull(cnt)
                  areahull = cv2.contourArea(hull)
                  areacnt = cv2.contourArea(cnt)
                  arearatio=((areahull-areacnt)/areacnt)*100
                  hull = cv2.convexHull(approx, returnPoints=False)
                  defects = cv2.convexityDefects(approx, hull)
                  l=0
                  
                  for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                    d=(2*ar)/a
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                    if angle <= 90 and d>30:
                       l += 1
                  l+=1
                  skin_d = cv2.bitwise_and(roiframe,roiframe, mask=mask)
                 
                  kp2, des2 = orb.detectAndCompute(skin_d, None)
                  matches = bf.match(des1, des2)
                  matches = sorted(matches, key = lambda x:x.distance)
                  matching_result = cv2.drawMatches(skin_d, kp1, img_hand_resiz1, kp2, matches[:100], None,flags=2)
                  
                 
                  if   (l==5 ) and len(matches)>=20 and upper>170:
                              contour,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                              for cnt in (contour ):
                               contourarea=cv2.contourArea(cnt)
                              cv2.putText(frame,'I see your hand ,now traking will begin :)',(10,50),cv2.cv2.FONT_HERSHEY_PLAIN,1.3, (0,255,255), 1,cv2.LINE_AA)
                              self.a+=1                            
                  else :
                      cv2.putText(frame,'I see no hand :(',(10,50),cv2.FONT_HERSHEY_PLAIN,1.3, (0,255,0), 1, cv2.LINE_AA)
                 except:
                    
                     pass
                 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                 self.msk = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(mask))
                 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                 self.canvas.create_image(200, h-240, image = self.msk, anchor= tkinter.NW)
                 self.canvas.create_text(282,450,fill="red",font=("Times 20 italic bold",9),text="Nb:put your hand in the box and make sure that you have a clear background for a better result")
                 
             self.window.after(self.delay, self.update)
             if (self.a>3):
                      self.window.destroy()
                  
          except:
       
              pass

         elif self.s==2:
           try: 
             ret,frame,mask,kernel = self.vid.get_frame(2)
             gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) 
             face_mask=mask.copy()
           
             if ret:
                try:
                        
                        faces = face_cascade.detectMultiScale(gray,1.2,7)
                        for (xf, yf, wf ,hf) in faces:
                                cv2.rectangle(mask, (xf-3,yf-10), (xf+wf+255 ,yf+hf+300), (0, 0 , 0), -1)
                                contour,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in (contour ):
                                  (x,y,w,h)=cv2.boundingRect(cnt)
                                  if (cv2.contourArea(cnt)>=contourarea-100) :
                                            if (h<=250):
                                              cv2.rectangle(frame, (x,y), (x+w,y+h-r), (0, 255 , 0), 2)
                                              roi=frame[y:y+h-r, x:x+w]
                                              roimask=mask[y:y+h-r, x:x+w]
                                              test_features= data_preper(roi,roimask)
                                              prediction=clf1.predict(test_features)
                                              p=categories[prediction[0]]
                                              cv2.putText(frame,'Your singe is : {}'.format(p),(10,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
                                              try:
                                               p=str.encode(p)
                                               UDPSock.sendto(p, addr)
                                              except :
                                                   raise
                                                   pass
                                              
                                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)))  
                        self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                except:
                     
                    pass
             self.window.after(self.delay, self.update)
            
           except:
               pass
        

class MyVideoCapture:
     
     def __init__(self,a,s,video_source=0):
       global h,w    
       if c==1:   
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  
       elif c==0:
         try :
            if UDPSock1.recv!=None:
              self.width,_ = UDPSock1.recvfrom(400)
              self.height,_ = UDPSock1.recvfrom(400)
              self.width=int(self.width.decode('utf-8'))
              self.height=int(self.height.decode('utf-8'))
             
              
         except:
               
               self.height=h
               self.width=w
               
       
       h,w=self.height,self.width
        
       if s==1:
        h=h+250
       elif s==2 and c==0:
         h=h-260  
     def get_frame(self,s):
       try:
            if c==1:
              if self.vid.isOpened():
                 ret, FrameRGB = self.vid.read()
            elif c==0:
                 
                if UDPSock1.recv!=None:
                  Framei=b''   
                  for i in range(0,4):
                    FrameRGB, addr1 = UDPSock1.recvfrom(70000)
                    FrameRGB=Framei+FrameRGB
                    Framei=FrameRGB
                  FrameRGB = cv2.imdecode(np.frombuffer(FrameRGB, np.uint8), -1)
                  FrameRGB.resize(480,640,3)
                  ret=1
            kernel = np.ones((3,3),np.uint8)     
            FrameBGR=treat_frame(FrameRGB)
            space=TreatHsvOrYcrcb(FrameBGR)
            y,cr,cb=cv2.split(space)
            if s==1:
                  roi=FrameBGR[100:300, 100:300]
                  roispace=space[100:300, 100:300]
                  cv2.rectangle(FrameBGR,(100,100),(300,300),(0,255,0),6)
                  pick_color(roi,cr)   
                  mask=cv2.inRange(cr[100:300, 100:300],lower,upper)
                  op= cv2.morphologyEx(mask, cv2.MORPH_OPEN,es,iterations=3)
                  mask = cv2.morphologyEx(op, cv2.MORPH_CLOSE,es,iterations=3)
                  mask = cv2.GaussianBlur(mask,(5,5),30)
                  mask=cv2.erode(mask,kernel,iterations = 2)
                  return (ret,FrameBGR,roi,mask,lower,upper)                
            elif s==2:  
                   mask1=cv2.inRange(cr,lower,upper)
                   op = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,es,iterations=3)
                   cls = cv2.morphologyEx(op, cv2.MORPH_CLOSE,es,iterations=3)
                   mask = cv2.GaussianBlur(cls,(5,5),30)
                 
                   return (ret,FrameBGR,mask,kernel)
       except:
                pass
              
       
     def __del__(self):
        if c==1:
         if self.vid.isOpened():
            self.vid.release()
  
     
   
def pick_color(roi,space_init): 
            global lower,upper,coord_x,coord_y,panel
            h,w=roi.shape[0],roi.shape[1]
            coord_x=int(150+w/2-50)
            coord_y=int(150+h/2-30)
            pixel = space_init[coord_y,coord_x]
            lower =  np.array([pixel - off1])
            upper =  np.array([pixel + off1])
 
         
def treat_frame (frame):
    filtred=cv2.GaussianBlur(frame,(3,3),0.2)
    b,g,r = cv2.split(filtred)
    b,g,r= clahe.apply(b),clahe.apply(g),clahe.apply(r)
    filtred = cv2.merge((b,g,r))
    return filtred

def TreatHsvOrYcrcb (space):
    space=cv2.cvtColor(space,cv2.COLOR_BGR2YCrCb)
    return space
    
def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(np.squeeze(contour), order=35, normalize=False)
    return coeffs.flatten()[1:]

def data_preper (frame,mask):

     frame=cv2.flip(frame,1)
     frame=cv2.resize(frame,(img_size,img_size))
     mask=cv2.flip(mask,1)
     mask=cv2.resize(mask,(img_size,img_size))
     data_test=[]
     contour,_=cv2.findContours(mask,cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
     #--------------------label-------------------------
     label_img= label(mask)
     #--------------fourier discriptot------------------------------------------
     for cnt in contour:
               fd=efd_feature(cnt)
     #----------------propeties--------------
     props = regionprops(label_img)
     huemoment=props[0].moments_hu
     huemoment=np.delete(huemoment,6)
     s=np.concatenate([huemoment,fd],axis=None)
     data_test.append(s)
     test_features=np.array(data_test)           
     return test_features
   
         
            
def main():
 try:
     choice(tkinter.Tk(), "make your choice")
     print ("Waiting to open the camera ...")
     App(tkinter.Tk(), "recognition",1)
     print ("hand data saved ...")
     App(tkinter.Tk(), "Tracking hand ",2)
 except :
      pass
 finally :
    UDPSock.close()
    UDPSock1.close()
    print("goodbye")
 

if __name__=='__main__':
    main()
