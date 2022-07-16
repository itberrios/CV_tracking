# Simple Implementation of the SORT algorithm

In this project multiple object tracking was implemented using YOLOv5 as the object detector and the Hungarian Algoirthm to correlate bounding boxes across frames. A Kalman Filter was used to extrapolate the centers of the bounding boxes across frames. This has generally lead to better tracking and has provided some mitigation for the object reidnetification that occurs occurs when objects are visually occluded.

The video below shows the comparison of a simple case where we attempt to detect and track vehicles on the road. The Hungarian only implementation is shown on top, while the Hunarian + Kalman Filter implementation is shown below. The vehicles are occasionally not detected by YOLO and the track IDs are updated, in this case the issue is mitigated by the Kalman Filter.

https://user-images.githubusercontent.com/60835780/179359520-586c8f22-ba03-4ff9-a9a5-6dde557ca4c5.mp4




This is a more complex scenario of vehicles and pedestrians. The Kalman Filter is not as robust in this scenario as some tracks tend to wander off without a match for a few frames (becoming false tracks). (top - Hungarian, bottom - Hungarian + Kalman Filter). Even though the Kalman Filter is able to extrapolate better, it leads to more false tracks as opposed to more reidentifications. This shows how complex the multiple object tracking problem can be, the kalman filter and tracking parameters can be tuned to this situation, but it may not be very helpful outside of this specific type of scenario.

https://user-images.githubusercontent.com/60835780/179359657-8a6dbf1c-42ee-4993-99b8-5e4bfeb55f1e.mp4



