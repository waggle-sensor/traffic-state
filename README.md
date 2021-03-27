## YOLOv4

Functions for YOLOv4 are in the foler `tool`.
The functions are called at `yolov4_plugin.ipynb` and `yolov4_plugin.py` in jupyter notebook and terminal respectively to detect vehicles.
The class information is in `detection` folder -- the coco classes.

## DeepSort
Functions for DeepSort that tracks vehicles are in `siamease_net.py` (main algorithm to track vehicles based on their features) and the folder `deep_sort`.
The functions are called at `deepsort.py` which is main class to run vehicle tracking.
Traffic state calculation are performed at `deepsort_plugin.ipynb` and `deepsort_plugin.py` in jupyter notebook and terminal respectively.
Traffic state calculation and vehicle tracking are based on the object detection results from YOLOv4, so that the `deeposrt_plugin.ipynb` and `deepsort_plugin.py` cannot be excuted by itself -- they requires input values that came out from `yolov4_plugin`.

## PyWaggle
We use the two methods (`YOLOv4` and `DeepSort`) to calculate traffic state.
And the calculation results are sent through `PyWaggle` to Cloud (for detail information see [PyWaggle](https://github.com/waggle-sensor/pywaggle)).
The `PyWaggle` is simply implemented in `pywaggle.ipynb` and `pywaggle.py`, and still testing the modules (3/27/2021).
