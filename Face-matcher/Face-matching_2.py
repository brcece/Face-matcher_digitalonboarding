import face_recognition
import numpy as np
import cv2
import time


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        return None  
    
    
    max_area = 0
    max_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            max_face = image[y:y + h, x:x + w]
   
    return max_face

def resize_image(image, desired_height=370):
    dimage = detect_faces(image)
    if dimage is not None:
        aspect_ratio = dimage.shape[1] / dimage.shape[0]
        desired_width = int(desired_height * aspect_ratio)
        dsize = (desired_width, desired_height)
        dimage = cv2.resize(dimage, dsize)
    return dimage

def encoding(image):
    try:
        image=resize_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    
        encoding = face_recognition.face_encodings(image)   

        return encoding
    except Exception as e:
        print(e)

def cos_sim(face,photo):
    try:
        face=encoding(face)
        photo=encoding(photo)
    
  
        a =sum(v1*v2 for v1,v2 in zip(face[0],photo[0]))

        b=sum([x**2 for x in face[0]])**(1/2)

        c=sum([x**2 for x in photo[0]])**(1/2)
    
    
        result=a/(b*c)
        arcresult=np.arccos(result)
    
        similarity_percentage = (1 - arcresult) * 100 
       

        if similarity_percentage <75:
            x= similarity_percentage*1.3143
            print(f"Similarity Percentage: {x:.2f}%")
        elif similarity_percentage >= 75 and similarity_percentage < 80:
            y= similarity_percentage*1.2405
            print(f"Similarity Percentage: {y:.2f}%")
            
        else:
            print(f"Similarity Percentage: {similarity_percentage:.2f}%")
    except Exception as e:
        print(e)
    

if __name__ == "__main__":
    start_time = time.time()
    face = cv2.imread("images/y_gorusme.jpg")
    photo = cv2.imread("images/y_kimlik.jpg")
    try:
        result=cos_sim(face,photo)
    except Exception as e:
        print(e)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)