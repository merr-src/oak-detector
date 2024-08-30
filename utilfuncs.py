import numpy as np
import cv2
import depthai as dai
import time
from pathlib import Path


# nn data, bounding box locations, are in <0..1> - should be normalized
def frame_norm(frame, bbox):
    normvals = np.full(len(bbox), frame.shape[0])
    normvals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normvals).astype(int)


def show_frame(name, frame, detections, labels):
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.imshow(name, frame)

    
def movdet(bghist, ksize, mmax):

    # ----OAK pipeline settings----
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # --------cam properties--------
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        bgsub = cv2.createBackgroundSubtractorKNN(bghist)
        start = time.monotonic()
        color2 = (255, 255, 255)
        count, movcount = 0, 0
        global to_nn

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                cv2.putText(frame, f"movedet fps: {count / (time.monotonic() - start):.2f}",
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2)
                cv2.imshow("rgb", frame)
                count += 1
                if frame is not None:
                    fg_mask = bgsub.apply(frame)
                    fg_mask_e = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
                    motion = cv2.findNonZero(fg_mask_e)
                    motframe = cv2.cvtColor(fg_mask_e, cv2.COLOR_GRAY2BGR)
                    cv2.imshow("motion", motframe)

                    if motion is not None:
                        movcount += 1
                    else:
                        if movcount != 0:
                            movcount -= 1

                    print(count, movcount)
                    if movcount == mmax:
                        to_nn = 'pass'
                        return to_nn

            if cv2.waitKey(1) == ord('q'):
                break


def detector(
        nodetmax,
        w, h,
        classes,
        coordinates,
        anchors,
        anchormasks,
        iouthresh,
        confthresh,
        labels,
        nnpath,
        savepath
        ):

    # --------init pipeline, define sources and outputs--------
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    # cam_rgb.preview.link(xout_rgb.input)
    xout_rgb.setStreamName("rgb")

    det_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    # --------cam settings--------
    cam_rgb.setPreviewSize(w, h)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)

    # --------network settings--------
    det_nn.setConfidenceThreshold(confthresh)
    det_nn.setNumClasses(classes)
    det_nn.setCoordinateSize(coordinates)
    det_nn.setAnchors(anchors)
    det_nn.setAnchorMasks(anchormasks)
    det_nn.setIouThreshold(iouthresh)
    det_nn.setBlobPath(nnpath)
    det_nn.setNumInferenceThreads(2)
    det_nn.input.setBlocking(False)

    # --------linking--------
    cam_rgb.preview.link(det_nn.input)
    det_nn.passthrough.link(xout_rgb.input)
    det_nn.out.link(xout_nn.input)

    syncNN = True  # sync outputs

    # --------init device--------
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_det = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        counter = 0
        nodetcount = 0
        detections = []
        start = time.monotonic()
        color2 = (255, 255, 255)
        global to_nn

        with open('results.csv', 'a') as f:
            f.write("file_name, time, cls, conf, xmin, ymin, xmax, ymax".upper() + '\n')

        # --------main loop--------
        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                cv2.putText(frame, f"NN fps: {counter / (time.monotonic() - start):.2f}",
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2)
                cv2.imshow("rgb", frame)
                counter += 1
                if frame is not None:
                    in_det = q_det.tryGet()

                    if in_det is not None:
                        detections = in_det.detections
                    if detections:
                        print('NN: detection')
                        show_frame("rgb", frame, detections, labels)
                        savename = str(counter) + '.jpg'
                        nodetcount = 0

                        # --------log detections--------
                        for detection in detections:
                            detdata = []
                            coords = frame_norm(
                                frame,
                                (
                                    detection.xmin,
                                    detection.ymin,
                                    detection.xmax,
                                    detection.ymax
                                )
                            )
                            conf = detection.confidence
                            timestamp = round(time.monotonic() - start, 5)
                            detdata.append(savename)
                            detdata.append(timestamp)
                            detdata.append(labels[detection.label])
                            detdata.append(round(conf, 2))
                            detdata.append(coords[0])
                            detdata.append(coords[1])
                            detdata.append(coords[2])
                            detdata.append(coords[3])

                            with open('results.csv', 'a') as ff:
                                ff.write(str(detdata)[1:-1] + '\n')

                            # --------save frame with detections--------
                            savepath2 = Path.cwd() / savepath / savename
                            print(savepath)
                            cv2.imwrite(str(savepath2), frame)
                    else:
                        nodetcount += 1
                        print(f'nodetcount: {nodetcount}, NN: no detections')
                    if nodetcount == nodetmax:
                        nodetcount = 0
                        to_nn = 'stop'
                        return to_nn

            if cv2.waitKey(1) == ord('q'):
                break
