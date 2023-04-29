import cv2
import numpy as np
from django.http import request
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing as mp

count_line_position = 550
min_width_rect = 100
min_height_rect = 100

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    length = int(width/2.5)
    
    detect = []
    offset = 6
    counter = 0

    while True:
        ret, frame = cap.read()
        
        #frame = cv2.resize(frame, (400, 500))

        if not ret:
            break

        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)

        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5,5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

        countershape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame, (0, length), (width, length), (255, 127, 0), 3)

        for (i, c) in enumerate(countershape):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w>=min_width_rect) and (h >= min_height_rect)
            if not validate_counter:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "vehicle "+str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            for (x, y) in detect:
                if y<(count_line_position + offset) and y>(count_line_position - offset):
                    counter += 1
                cv2.line(frame, (0, length), (width, length), (0, 127, 255), 3)
                detect.remove((x, y))
                print('vehicle counter' + str(counter))
        cv2.putText(frame, "vehicle counter:"+str(counter),(450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        cv2.namedWindow('video original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('video original', 500, 400)
     
        cv2.imshow('video original', frame)

        if cv2.waitKey(1) == 13:
            break
    cv2.destroyAllWindows()
    cap.release()

def get_uploaded_videos(request):
    video_files = []
    if request.method == 'POST' and request.FILES:
        for file in request.FILES.getlist('video'):
            # Save the file temporarily
            with open(file.temporary_file_path(), 'wb+') as f:
                f.write(file.read())
            # Add the file name to the list
            video_files.append(file.name)
    return video_files

if __name__ == '__main__':
    # List of video files
    video_files = ['cars2.mp4', 'cars2.mp4']

    #video_files = get_uploaded_videos()

    # Number of processes to create
    num_processes = len(video_files)

    # Create a multiprocessing pool
    pool = mp.Pool(processes=num_processes)

    # Process each video file using the process_video function
    for video_file in video_files:
        pool.apply_async(detect_objects, args=(video_file,))

    # Wait for all processes to finish
    pool.close()
    pool.join()
