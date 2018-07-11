import numpy as np
import argparse
import cv2

# constructing the argument parse and parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
	help="path of input image")
parser.add_argument("-p", "--prototxt", required=True,
	help="path of Caffe 'deploy' prototxt file")
parser.add_argument("-m", "--model", required=True,
	help="path of Caffe pre-trained model")
parser.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detection")
args = vars(parser.parse_args())



# initialize the list of class labels MobileNet SSD was trained to detect, after that generate  bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 200, size=(len(CLASSES), 3))



# load our serialized model from disk
print("model is loading ......")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it (note: normalization is done via the authors of the MobileNet SSD implementation)
read_image = cv2.imread(args["image"])
(h, w) = read_image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(read_image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detection and predictions
print("computing object detect...")
network.setInput(blob)
detect = network.forward()

# loop over the detect
for i in np.arange(0, detect.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detect[0, 0, i, 2]

	# filter out weak detect by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detect`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detect[0, 0, i, 1])
		box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(read_image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(read_image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# showing of  the output image
cv2.imshow("Output", read_image)
cv2.waitKey(0)
