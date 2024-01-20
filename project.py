
import cv2
import math
from ultralytics import YOLO

cam=cv2.VideoCapture('media.mp4')
model = YOLO("yolov8n.pt")

count=0
frame_count=0
last={}
tracking_obj={}
up={}
track_id=0

def calculate_iou(box1, box2):
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    intersection_area = max(0, x2_int - x1_int + 1) * max(0, y2_int - y1_int + 1)
    intersection_area1 = max(0, x1_int - x2_int + 1) * max(0, y1_int - y2_int + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    box1_area1 = (box1[0] - box1[2] + 1) * (box1[1] - box1[3] + 1)
    box2_area1 = (box2[2] - box2[2] + 1) * (box2[1] - box2[3] + 1)

    iou = (intersection_area / float(box1_area + box2_area - intersection_area))*100
    iou1 = (intersection_area1 / float(box1_area1 + box2_area1 - intersection_area1)) * 100
    return max(iou,iou1)
pt_previous=[]
while True:
    ret,frame=cam.read()
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(int(fps))
    count+=1
    if not ret:
        break
    results = model(frame, stream=True,)

    pt_current=[]

    #For box and center point
    for r in results:


        boxes = r.boxes

        overs=[]
        over=[]
        pt=[]

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            classs=box.cls[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classs=int(classs)


            if classs in [2,3,4,7,8]:

                over=[x1,y1,x2,y2]
                res=[0.000000000000000000001]
                if len(overs) >=1:
                    for i in overs:
                        res.append(calculate_iou(i,over))
                overs.append(over)
                x3=int((x1+x2)/2)
                y3 = int((y1 + y2) / 2)

                if max(res)<=30:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.circle(frame,(x3,y3),3,(0,0,255))
                    pt.append([x3,y3] )
        pt_current=pt

    print(pt_previous )
    print(pt_current)
    pt_current1=pt_current.copy()
    #for initial id initialisation
    if count<=1:


        for pt1 in pt_current:
            tracking_obj[track_id] = pt1
            track_id += 1

    #for the rest frames
    else:

        up=tracking_obj.copy()
        #keep on Tracking the object
        for pt1 in pt_current:
            no_match = 0
            for ob_id,pt2 in tracking_obj.items():
                distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                if distance <= 35:
                    print(ob_id)
                    no_match =1
                    tracking_obj[ob_id] = pt1
                    print("len of up",len(up))
                    print(up)
                    if len(up)>=1:

                        del up[ob_id]

            if no_match==0 :
                one_no_match=0
                print(last.items())
                for ob_id1,itemdict in last.items():
                    for ob_id,pt2 in itemdict.items():
                        print(pt2)
                        distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                        if distance <= 30:
                            one_no_match=1
                            tracking_obj[ob_id] = pt1

                if one_no_match==0:
                    tracking_obj[track_id] = pt1
                    track_id+=1

    for i,j in up.items():
        del tracking_obj[i]


    for ob_id, pt1 in tracking_obj.items():

        cv2.putText(frame, str(ob_id), (pt1[0], pt1[1] - 7), 0, 1, (0, 0, 255))

    last[frame_count] = up
    if len(last)>5:
        last.pop(list(last.keys())[0])

    pt_previous = pt_current1
    cv2.imshow("Frames", frame)


    frame_count+=1

    cv2.waitKey(1)



