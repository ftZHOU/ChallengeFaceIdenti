import sys
import os
import dlib
import glob
from skimage import io 
import numpy



if len(sys.argv) != 5:
    print("call this program in the right order (python3 test.py shape_predictor_68_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat testFace.jpg)")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3] # any person 
img_path = sys.argv[4]  #the person compared

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []

# take average of each person 
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
	print("Processing file: {}".format(f))
	img = io.imread(f)

	dets = detector(img, 1)
	print("Number of faces detected: {}".format(len(dets)))

	for k, d in enumerate(dets):
		shape = sp(img, d)
		face_descriptor = facerec.compute_face_descriptor(img, shape)
		v = numpy.array(face_descriptor) 
		descriptors.append(v)

person = numpy.array(numpy.mean(descriptors,axis=0))  #calculate the average of this person's vector 

# take the picture tested 
img = io.imread(img_path)
dets = detector(img, 1)
for k, d in enumerate(dets):
	shape = sp(img, d)
	face_descriptor = facerec.compute_face_descriptor(img, shape)
	tester = numpy.array(face_descriptor) 

# calculate the distance 

distance = numpy.linalg.norm(person-tester)
print(distance)

# if the distance is lower than 0.6 than we have found the right face 




























