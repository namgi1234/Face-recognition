import cv2
import numpy as np
from roboflow import Roboflow
import tkinter


#분류기
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#카메라 세팅
capture = cv2.VideoCapture(0) #초기화, 카메라 번호 (0:내장, 1:외장)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #4

#console
face_id = input('\n 사용자 이름을 입력해주세요. ==>') #사용자 id 입력 받기
print("\n [INFO] Initializing face capture. Look the CAMERA and wait")

count = 0 #데이터로 저장할 얼굴의 수

#영상 처리 및 출력
while True:
    ret, frame = capture.read() #카메라 상태, 프레임
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백으로
    faces = faceCascade.detectMultiScale(gray, 1.1, 10)
    
    
    #얼굴에 대해 rectangle 출력
    if len(faces):
        for (x,y,w,h) in faces: #(x,y):얼굴의 좌상단 위치, (w,h):가로,세로
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #이미지,좌상단좌표, 우하단좌표, 색상, 선두께)
            count += 1
            cv2.imwrite("C:/Users/JEW0311/Desktop/CV/dataset/data."+str(face_id)+'.'+str(count)+".jpg", gray[y:y+h, x:x+w])
        
    cv2.imshow('image',frame)
    
    #종료조건
    if cv2.waitKey(1)>0: break
    elif count>=100:
        # window=tkinter.Tk()
        # window.title("업로드 중...")
        # window.geometry("640x400+100+100")
        # window.resizable(False, False)
        # label=tkinter.Label(window, text="로딩중.....")
        # label.pack()
        # window.mainloop()
        rf = Roboflow(api_key="roboflow api key")

        # Retrieve your current workspace and project name
        print(rf.workspace())

        # Specify the project for upload
        # let's you have a project at https://app.roboflow.com/my-workspace/my-project
        workspaceId = 'shiftai-9gxfr'
        projectId = 'afafafa'
        project = rf.workspace(workspaceId).project(projectId)

        # Upload the image to your project
        #project.upload(path)

        """
        Optional Parameters:
        - num_retry_uploads: Number of retries for uploading the image in case of failure.
        - batch_name: Upload the image to a specific batch.
        - split: Upload the image to a specific split.
        - tag: Store metadata as a tag on the image.
        - sequence_number: [Optional] If you want to keep the order of your images in the dataset, pass sequence_number and sequence_size..
        - sequence_size: [Optional] The total number of images in the sequence. Defaults to 100,000 if not set.
        """
        for i in range(1,100 + 1):
            path = 'C:/Users/JEW0311/Desktop/CV/dataset/data.1.'+str(i)+'.jpg'
            project.upload(
                image_path=path,
                batch_name="MyFace",
                split="face",
                num_retry_uploads=3,
                tag="Name",
                sequence_number=99,
                sequence_size=100
            )
            print(i,"/ 100")
        break
capture.release() #메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 닫기

