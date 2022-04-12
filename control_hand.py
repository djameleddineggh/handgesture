import sys
from socket import *
import cv2
import RPi.GPIO as GPIO
import numpy as np
import time
import threading
import tkinter 
from tkinter import *
from tkinter import ttk
from tkinter.ttk import *

GPIO.setwarnings(False)
#-------------------default angels------------
d1h,d1l=7.4,10
d2h,d2l=8.5,5.7
d3h,d3l=13,9
d4h,d4l=11.7,6.5
d5h,d5l=11.7,5.7
#---------------init servo------
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.OUT)
servo1 = GPIO.PWM(7,50) # pin 11 for servo1
GPIO.setup(11,GPIO.OUT)
servo2 = GPIO.PWM(11,50) # pin 12 for servo2
GPIO.setup(13,GPIO.OUT)
servo3 = GPIO.PWM(13,50) # pin 11 for servo3
GPIO.setup(15,GPIO.OUT)
servo4 = GPIO.PWM(15,50) # pin 12 for servo4
GPIO.setup(16,GPIO.OUT)
servo5 = GPIO.PWM(16,50) # pin 11 for servo5
d1,d1p,d2,d2p,d3,d3p,d4,d4p,d5,pinc=0,0,0,0,0,0,0,0,0,0
servo1.start(0)
servo2.start(0)
servo3.start(0)
servo4.start(0)
servo5.start(0)

do=0
ipraspi="192.168.43.74"
ippc="192.168.43.91"
addr = (ippc, 12000)
UDPSock = socket(AF_INET, SOCK_DGRAM)#pc
addr1 = (ipraspi, 15000) #raspi
UDPSock1 = socket(AF_INET, SOCK_DGRAM)

control=[d1l,d2l,d3l,d4l,d5l]
c=-1

class choice:
     
     def __init__(self,window, window_title):
      self.window = window
      self.window.title(window_title)
      
      self.style1 = ttk.Style()
      self.style1.theme_use('alt')
    
      self.style1.configure('B1.TButton',
        highlightthickness='20',
        font=('Arial', 12, 'bold'))
      self.style1.map("B1.TButton",
       foreground=[('pressed', 'red'),('!disabled','yellow') ],
       background=[('pressed',  'black'), ('!disabled','black')],           
       )
      
      self.btn_choice1=ttk.Button(window, text="pi camera",width=20,command=self.raspberry,style="B1.TButton")
      self.btn_choice1.pack(side=tkinter.LEFT)
      
      self.style2 = ttk.Style()
      self.style1.theme_use('alt')
      self.style1.configure('B2.TButton',
        highlightthickness='20',
        font=('Arial', 12, 'bold'))
      self.style1.map("B2.TButton",
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

def comand():
    
    while True:
 
      global d1,d1p,d2,d2p,d3,d3p,d4,d4p,d5,pinc,w
      if w==True:
       

       servo1.ChangeDutyCycle(control[0])
      
       servo2.ChangeDutyCycle(control[1])
     
       servo3.ChangeDutyCycle(control[2])
       
       servo4.ChangeDutyCycle(control[3])
     
       servo5.ChangeDutyCycle(control[4])
       time.sleep(0.5)
       servo1.ChangeDutyCycle(0)
       servo2.ChangeDutyCycle(0)
       servo3.ChangeDutyCycle(0)
       servo4.ChangeDutyCycle(0)
       servo5.ChangeDutyCycle(0)
       time.sleep(1)
       d1,d1p,d2,d2p,d3,d3p,d4,d4p,d5,pinc=0,0,0,0,0,0,0,0,0,0
       w=False
       time.sleep(1)

       

UDPSock1.bind(addr1)
def recive1(UDPSock):
    global d1,d1p,d2,d2p,d3,d3p,d4,d4p,d5,pinc,w,control
    
    while True:

         if UDPSock.recv!=None:
            p,_= UDPSock.recvfrom(100000)
            p=p.decode('utf-8')
   
            if p=='digit1':
                d1=d1+1
                if d1>=do :
                 control=[d1l,d2h,d3l,d4l,d5l]
                 w=True    
            elif p=='digit1p' :
                d1p=d1p+1
                if d1p>=do :
                 control=[d1h,d2l,d3l,d4l,d5l]
                 w=True
              
            elif p=='digit2':
                d2=d2+1
                if d2>=do:
                 control=[d1l,d2h,d3h,d4l,d5l]
                 w=True
            elif p=='digit2p':
                d2p=d2p+1
                if d2p>=do:
                 control=[d1h,d2h,d3l,d4l,d5l]
                 w=True
           
            elif p=='digit3':
                d3=d3+1
                if d3>=do:
                 control=[d1l,d2h,d3h,d4h,d5l]
                 w=True
           

            elif p=='digit3p':
                d3p=d3p+1
                if d3p>=do:
                 control=[d1h,d2h,d3h,d4l,d5l]
                 w=True
                 
            elif p=='digit4':
                d4=d4+1
                if d4>=do:
                 control=[d1l,d2h,d3h,d4h,d5h]
                 w=True
                
            elif p=='digit4p':
                d4p=d4p+1
                if d4p>=do:
                 control=[d1h,d2h,d3h,d4h,d5l]
                 w=True
               
            elif p=='digit5':
                d5=d5+1
                if d5>=do: 
                  control=[d1h,d2h,d3h,d4h,d5h]
                  w=True
                
            elif p=='pince':
                pinc=pinc+1
                if pinc>=do:
                  control=[d1l,d2l,d3h,d4h,d5h]
                  w=True
              

def send(UDPSock,cap):
  fi=b''
  while True:  
    _,frame=cap.read()
    frame.resize()
    f=cv2.imencode('.jpg',frame)[1].tostring()
    for i in range(0,4):
      fi1=f[len(fi):65000]
      UDPSock.sendto(fi1, addr)
      fi=fi1
    if cv2.waitKey(1)==ord(" "):
        break
  cap.release()      
  cv2.destroyAllWindows()
  
choice(tkinter.Tk(), "make your choice")  
if c==0:
 cap=cv2.VideoCapture(0)
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 w,h=str.encode(str(height)),str.encode(str(width ))
 UDPSock.sendto(w, addr)
 UDPSock.sendto(h, addr)           
 t1= threading.Thread(target=send, args=[UDPSock,cap])
 t1.daemon = True
 t1.start()
elif c==1:
    pass
t2 = threading.Thread(target=recive1, args=[UDPSock1])
t2.daemon = True
t2.start()

w=True
t3 = threading.Thread(target=comand)
t3.daemon = True
t3.start()
try :
 while True:

    time.sleep(0)
except :
     pass
finally :
 UDPSock.close()
 UDPSock1.close() 
 servo1.stop()
 servo2.stop()
 servo3.stop()
 servo4.stop()
 servo5.stop() 
 GPIO.cleanup()
 print("goodbye")
 
 
