#----IMPORTS-----
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
import math

#----OBJECT DETECTION MODELS----
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='model/P1_V2_openvino_2022.1_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='model/P1_V2.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# -- PIPELINE FROM OAK-D LITE --
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# ---- Pressure Gauge ----
# Calibration table mapping angles (degrees) to psi readings
calibration_table = {
    240:50, #3.5
    247.5: 58, #4
    255:65, #4.5
    262.5:73, #5
    270:80, #5.5
    277.5: 87, #6
    285: 94, #6.5
    292.5:102, #7
    300: 10.9, #7.5
    307.5: 116, #8
    315: 123, #8.5
    322.5: 131, #9
    330: 138, #9.5
    337.5: 145, #10

}

def angle_to_psi(angle, calibration_table):
    angles = list(calibration_table.keys())
    psi_values = list(calibration_table.values())
    
    if angle in angles:
        return calibration_table[angle]
    
    # Find the two closest calibrated angles
    angles.sort()
    for i in range(1, len(angles)):
        if angle < angles[i]:
            lower_angle, upper_angle = angles[i - 1], angles[i]
            lower_psi, upper_psi = psi_values[i - 1], psi_values[i]
            break
    else:
        # Angle is larger than all calibrated values, use the highest
        return psi_values[-1]

    # Perform linear interpolation
    psi = np.interp(angle, [lower_angle, upper_angle], [lower_psi, upper_psi])
    return psi

def pressure_gauge_1(detections):
    try:
        print(detections)
        centre_target = "centre"
        tip_target = "tip"

        
        center_coordinates = None
        for detection in detections:
            if labels[detection.label] == centre_target:
                print('hi')
                # Extract the coordinates of the bounding box (xmin, ymin, xmax, ymax)
                center_coordinates = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                
            if labels[detection.label] == tip_target:
                print('ohla')
                tip_coordinates = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                
        dy = tip_coordinates[1] - center_coordinates[1]
        dx = tip_coordinates[0] - center_coordinates[0]
        theta = math.atan2(dy, dx)  # Calculate the angle in radians

        # Convert the angle to degrees and adjust the starting point as needed
        theta_degrees = math.degrees(theta)
        print(f'Gauge Reading: {theta_degrees} degrees')
        
        if theta_degrees < 0:
            # Adjust the angle for negative values
            theta_degrees += 360
            #theta_degrees = abs(theta_degrees)

        # Perform any additional adjustments if required (e.g., handling negative angles)
        print(f'Gauge Reading: {theta_degrees} degrees')
        
        psi_reading = angle_to_psi(theta_degrees, calibration_table)
        print(psi_reading)
        
        print(f'Gauge Reading: {psi_reading} psi')
        return psi_reading

            
    except Exception as e:
        print()


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    #Calibrate Camera
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
    dist = calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)

    np.save("calibration_matrix", intrinsics)
    np.save("distortion_coefficients", dist)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Check if the label is for the gauge object (adjust this condition accordingly)
            if "centre" in labels and "tip" in labels:
                print('here')
                # Calculate the angle and update the gauge reading
                psi_reading=pressure_gauge_1(detections)
                cv2.putText(frame, f'Gauge Reading: {psi_reading} psi', (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            # Where detections are (labels[detection.label])
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame, detections)



        key = cv2.waitKey(1)
        if key == ord('q'):
            break  # Exit the loop if 'q' is pressed