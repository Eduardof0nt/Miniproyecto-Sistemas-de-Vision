import tkinter as tk
from tkinter import *
import tkinter.filedialog

import matplotlib
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import numpy as np

from skimage import io
from skimage.color import rgb2grey
from skimage.filters import gaussian
import skimage.feature

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2


root = tk.Tk() #La "Raiz" de la app con tkinter
apps=[]

def agregarimagen(): #Funcion para abrir archivos
    global mensaje1, mensaje2, mensaje3
    mensaje1['text'] = ''
    mensaje2['text'] = ''
    mensaje3['text'] = ''
    plt.close('all')
    plt.show(block = False)
    filename = tkinter.filedialog.askopenfilename(initialdir="/", title="Seleccione la imagen",
    filetypes=(("JPG","*.jpg"), ("JPeG","*.jpeg"), ("PNG", "*.png")))
    apps.append(filename)
    
    image = io.imread(filename, as_gray=False)[:,:,0:3]
    image = imutils.resize(image, height=1000)

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.set_title("Imagen Original")
    axs.axis('off')
    axs.imshow(image,cmap = None)
    plt.show(block = False)
    
    imageGS = (rgb2grey(image)*255).astype('uint8')
    imageGS = gaussian(imageGS, sigma=1)

    # fig1, axs1 = plt.subplots(1, 1, figsize=(10, 6.5))
    # axs1.set_title("Imagen en Tonos de Gris")
    # axs1.imshow(imageGS,cmap = plt.cm.gray)

    img = image.copy()
    s = img.shape

    # ----------- Segmentación por color -----------
    img2 = np.zeros_like(img)

    color = [220,70,70]
    # color = [240,240,160]

    color = np.array([[color]*s[1]]*s[0])
    colorComparation = image-color
    colorComparation = np.power(colorComparation,2)
    colorComparation = np.sqrt(np.sum(colorComparation,2))
    colorComparation = (colorComparation < 45).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    colorComparation = cv2.dilate(colorComparation, kernel)

    cnts = cv2.findContours(colorComparation.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # fig3, axs3 = plt.subplots(1, 1, figsize=(10, 6.5))
    # axs3.set_title("Imagen Con Segmentación Por Color")
    # axs3.imshow(colorComparation,cmap = plt.cm.gray)

    digitCnt = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 30 and h > 50 and 1.5*w < h :
            peri = cv2.arcLength(c, True)
            digitCnt.append(cv2.approxPolyDP(c, 0.1 * peri, True))
            cv2.fillPoly(img2, pts =[c], color=(255,255,255))

    # fig3, axs3 = plt.subplots(1, 1, figsize=(10, 6.5))
    # axs3.set_title("Bordes De Imagen Con Segmentación Por Color")
    # axs3.imshow(img2,cmap = plt.cm.gray)


    # ----------- Segmentación por forma -----------

    thresh = ((imageGS < 0.08)*255).astype('uint8')

    # fig3, axs3 = plt.subplots(1, 1, figsize=(10, 6.5))
    # axs3.set_title("Binarización Para Segmentaión Por Forma")
    # axs3.imshow(thresh,cmap = plt.cm.gray)

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = []
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        (x, y, w, h) = cv2.boundingRect(c)
        if len(approx) == 4 and h < 0.95*s[1]:
            displayCnt.append(approx)
            cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

    # fig3, axs3 = plt.subplots(1, 1, figsize=(10, 6.5))
    # axs3.set_title("Imagen Bordes Rectángulo")
    # axs3.imshow(img,cmap = plt.cm.gray)


    # ----------- Combinación de segmentaciones anteriores -----------

    displayCnt = sorted(displayCnt, key=cv2.contourArea, reverse=True)
    display = None

    #Revisa cual de los contronos rectangulares contiene los contornos del color deseado

    possibleDisplays = []
    for c in displayCnt:
        weight = 0
        for c2 in digitCnt:
            for c3 in c2:
                if cv2.pointPolygonTest(c, (c3[0,0],c3[0,1]), False) >= 0:
                    weight += 1
        possibleDisplays.append((weight,c))

    possibleDisplays.sort(key=lambda x: x[0], reverse=True)
    display = possibleDisplays[0][1] 

    warped = four_point_transform(imageGS, display.reshape(4, 2))
    output = four_point_transform(image, display.reshape(4, 2))

    img = image.copy()

    (x, y, w, h) = cv2.boundingRect(display)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    mensaje1['text'] = 'Posición del display: x:{0}, y:{1}'.format(x+w//2,y+h//2,w,h)
    cv2.putText(img, str('x:{0}, y:{1}, w:{2}, h:{3}'.format(x+w//2,y+h//2,w,h)), (x-15, y-15),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.set_title("Ubicación Del Display En La Imagen")
    axs.axis('off')
    axs.imshow(img,cmap = plt.cm.gray)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6.5))
    axs.set_title("Extracto Del Display Para Procesar")
    axs.axis('off')
    axs.imshow(output,cmap = plt.cm.gray)

    warped = (warped*255).astype('uint8')

    thresh = cv2.threshold(warped.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) < 128

    imageBorders = thresh.astype('uint8')*255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    imageBorders = cv2.dilate(imageBorders, kernel)

    imageBorders2 = imageBorders.copy()
    cnts = cv2.findContours(imageBorders2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        cv2.fillPoly(imageBorders2, pts=[c], color=(255,255,255))
    plt.show(block = False)
    
    
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 1, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    img = output.copy()
    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(imageBorders2.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    bounds = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is sufficiently large, it must be a digit
        if w >= 15 and (h >= 50 and h <= 200):
            digitCnts.append(c)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
    def a(c):
        (x, y, w, h) = cv2.boundingRect(c)
        return -w*h

    digitCnts = np.array(sorted(digitCnts,key=lambda c: a(c)),dtype=object)
    
    
    
    digitCnts2 = contours.sort_contours(digitCnts,method="left-to-right")[0]
    
    # fig3, axs3 = plt.subplots(1, 1, figsize=(10, 6.5))
    # axs3.set_title("Digitos Encontrados")
    # axs3.imshow(digitCnts2,cmap = plt.cm.gray)
    # plt.show(block = False)
    
    digits = []

    damagedSegment = False

    # loop over each of the digits
    for c in digitCnts2:
        # extract the digit ROI
        (xm, ym, wm, hm) = cv2.boundingRect(digitCnts[0])
        print((xm, ym, wm, hm))
        (x, y, w, h) = cv2.boundingRect(c)
        if (wm*hm/1.5 > w*h):
            (x, y, w, h) = (x+w-wm, ym, wm, hm)
            
        roi = imageBorders[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)
        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB].astype('uint8')

            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if total / float(area) > 0.45:
                on[i]= 1

        # lookup the digit and draw it on the image
        try:
            # print(tuple(on))
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append((digit,tuple(on)))
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(output, str(digit), (x + w//4, y + h//2),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        except:
            damagedSegment = True
            digits.append(('X',tuple(on)))
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(output, str('X'), (x + w//4, y + h//2),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6.5))
    axs.set_title("Imagen con los Digitos Detectados")
    axs.axis('off')
    axs.imshow(output,cmap = plt.cm.gray)
    plt.show(block = False)

    # (arr, aizq, ader, med, abizq, abjder, ab)

    #Possible correction results
    possibleResult = []
    ds = []
    for i in range(0,len(digits)):
        if digits[i][0] == 'X':
            ds.append(i+1)
            weights = []
            damaged = np.array(digits[i][1])
            for key in DIGITS_LOOKUP:
                keyA = np.array(key)
                w = np.sum(np.abs(keyA-damaged))
                for i in range(0,len(keyA)):
                    if keyA[i] == 1 and damaged[i] == 0:
                        w += 5
                weights.append((w,key))
            weights.sort(key=lambda x: x[0], reverse=False)
            possibleResult.append(DIGITS_LOOKUP[weights[0][1]])
        else:
            possibleResult.append(digits[i][0])
            
    result = ''
    for i in range(0,len(possibleResult)):
        result += str(possibleResult[i])
        if len(possibleResult)-i == 3:
            result += ':'

    msg = 'El posible resultado de la lectura de los dígitos es el siguiente: '
    if damagedSegment:
        damagesSegmentsString = 'Segmento {0}'.format(ds[0])
        for i in range(1, len(ds)):
            damagesSegmentsString += ', '
            damagesSegmentsString += 'segmento {0}'.format(ds[i])
        mensaje2['text'] = 'Se encontraron los siguientes segmentos dañados: ' + damagesSegmentsString + '\n'
    else:
        msg = 'Se leyeron los siguientes dígitos: '
    mensaje3['text'] =  msg + result
    
    root.lift()

    


    
root.title("Miniproyecto Ricardo_Díaz-Eduardo_Font") #Titulo de la app
root.resizable(2,1) #Se puede hacer más grande o pequeña tanto en x como en y
root.geometry("650x650") #Este es el tamaño que va a tener cuando empieza la app
root.config(bg="#263D42")
marco = tk.Frame(root, bg="white")
marco.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1,)


abrirarchivo = tk.Button(marco, text="Abrir Archivo", padx=10, pady=5, fg="black", bg="#263D42", command=agregarimagen)
abrirarchivo.place(x=200, y=10)

mensaje1 = Label(marco, bg='#FFFFFF')
mensaje1.place(x=80, y=60)

mensaje2 = Message(marco, width=400, fg='#E30303', bg='#FFFFFF')
mensaje2.place(x=80, y=100)

mensaje3 = Label(marco, bg='#FFFFFF')
mensaje3.place(x=80, y=160)

root.mainloop()