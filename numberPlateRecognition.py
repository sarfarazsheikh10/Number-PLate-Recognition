import streamlit as st
import cv2, os
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd= r'C:\Program Files\Tesseract-OCR\tesseract.exe'



class NumberPlateRecognizer:


    @st.cache(allow_output_mutation=True)
    def load_network(self, config_path, weights_path):
        
        # Load the network
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Determine output layer names
        output_layer_names= net.getUnconnectedOutLayersNames()

        return net, output_layer_names

    def processImage(self, image):

        # Initialize parameters
        confThreshold = 0.5  #Confidence threshold
        nmsThreshold = 0.4   #Non-maximum suppression threshold
        inpWidth = 416       #Width of network's input image
        inpHeight = 416      #Height of network's input image

        config_path= os.path.join('config', 'yolov4-custom.cfg')
        weights_path= os.path.join('weights', 'custom.weights')

        net, output_layer_names= self.load_network(config_path, weights_path)

        # Create a 4D blob from image.
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (inpWidth, inpHeight), [0,0,0], crop = False)
        
        #set the input blob for the neural network
        net.setInput(blob)

        # forward pass image blob through the network
        output = net.forward(output_layer_names)

        # get the detected boxes
        boxes, classIDs, confidences = self.getBoxes(image, output, confThreshold, nmsThreshold)
        if ( len(boxes)==0 ):
            return image

        # get the alphanumerics out of the detected boxes 
        plateNumbers = self.recognizePlate(boxes, image)

        # draw number plates
        outputImage = self.drawPlates(image, boxes, confidences, plateNumbers)

        return outputImage

    def processVideo(self, vid):
        
        cap = cv2.VideoCapture(vid)

        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fourcc = 0x00000021
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        writer = cv2.VideoWriter("data/detections/detected_video.mp4", fourcc, fps, (frame.shape[1], frame.shape[0]))
        

        count=0        

        message = st.empty()
        progress_bar = st.progress(0)
        while cap.isOpened():
            
            count+=1
            ret, frame = cap.read()
            
            if not ret:
                break
            
            message.markdown(f'Processing frames ({count}/{total_frames}) ...')
            progress_bar.progress(count/total_frames)

            outputFrame = self.processImage( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
            writer.write(cv2.cvtColor(outputFrame, cv2.COLOR_BGR2RGB))
            

        message.empty()
        progress_bar.empty()

        # Release everything if job is finished
        cap.release()
        writer.release()
    
    def getBoxes(self, image, outputs, confThreshold, nmsThreshold):
    
        '''
        Removes low confidence bounding boxes and
        performs non-maxima suppression
        '''


        # Image width & height
        (H, W) = image.shape[:2] 
        
    
        # Scan through all the bounding boxes obtained from the networks output and keep only the
        # ones with high confidence scores
        classIds = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:

                scores = detection[5:]                          # predicted confidence scores for all classes
                classId = np.argmax(scores)                     # get the index of the highest confidence score
                confidence = scores[classId]                    # store the highest confidence score

                if confidence > confThreshold:

                    box = detection[:4] * np.array([W, H, W, H])                # get bounding-box location
                    (centerX, centerY, width, height) = box.astype('int') 

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)
                    
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    
        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        # get boxes, classes and confidences obtained after NMS
        boxes = [ boxes[index] for index in indices ]
        classIds = [ classIds[index] for index in indices ]
        confidences = [ confidences[index] for index in indices ]

        # return detected boxes
        return boxes, classIds, confidences

    def recognizePlate(self, boxes, image):
        '''
        This function passes each element in boxes (number plate)
        to the extractNumbers function, the latter function then gets the alphanumerics out of it.

        '''

        # empty list to contain number plate string
        plateNumbers=[]

        for box in boxes:
            (x,y,w,h)= box
            #print("Box: ", x,y,w,h)
            img= image[y:y+h, x:x+w].copy()
            plateNumbers.append( self.extractNumbers(img) )
        
        return plateNumbers

    def extractNumbers(self, image):
        '''
        This function takes the image of a number plate and returns the alphanumerics in string format
        '''
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            gray = cv2.medianBlur(gray, 3)
            
            # perform otsu thresh (using binary inverse since opencv contours work better with white text)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

            

            # apply dilation
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # create the structuring element
            dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

            # find contours
            try:
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            
            plate_num = ""
            # loop through contours and find letters in license plate
            for cnt in sorted_contours:
                x,y,w,h = cv2.boundingRect(cnt)
                height, width = image.shape[:2]
                
                # if height of box is not a quarter of total height then skip
                if float(h)<height/6 : continue
                
                # if height to width ratio is less than 1.5 skip
                ratio = h / float(w)
                if ratio < 1.5: continue
                
                # if width is not more than 25 pixels skip
                if width / float(w) > 15: continue

                # if area is less than 100 pixels skip
                area = h * w
                if area < 100: continue

                # draw the rectangle
                roi = thresh[y-5:y+h+5, x-5:x+w+5]
                roi = cv2.bitwise_not(roi)
                roi = cv2.medianBlur(roi, 5)
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                plate_num += text
            
            #print("PLate number: ",repr(plate_num))
            plate_num = ''.join([ char for char in plate_num if char.isalnum()])
            
            return plate_num

        except:
            return ''

    def drawPlates(self, image, boxes, confidences, plateNumbers):

        img = image.copy()

        for i in range(len(boxes)):
            x,y,w,h = boxes[i]

            # get plate number
            text= plateNumbers[i]
            if not text:
                continue

            # draw bounding box for number plate
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (21,76,121), 3)

            
            #draw rectangle for text
            (tw,th), basline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 2)
            img = cv2.rectangle(img, (x,y), (x+tw,y-th-4), (21,76,121), -1)

            # draw alphanumerics
            img = cv2.putText(img, text, (x+1,y-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2)
            
        return img                