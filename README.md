## YOLOv4

- Functions for YOLOv4 are in the foler `tool`. <br/>
- The functions are called at `yolov4_plugin.ipynb` and `yolov4_plugin.py` in jupyter notebook and terminal respectively to detect vehicles. <br/>
- The class information is in `detection` folder -- the coco classes.

## DeepSort
- Functions for DeepSort that tracks vehicles are in `siamease_net.py` (main algorithm to track vehicles based on their features) and the folder `deep_sort`. <br/>
- The functions are called at `deepsort.py` which is main class to run vehicle tracking. <br/>
- Traffic state calculation are performed at `deepsort_plugin.ipynb` and `deepsort_plugin.py` in jupyter notebook and terminal respectively. <br/>
- Traffic state calculation and vehicle tracking are based on the object detection results from YOLOv4, so that the `deeposrt_plugin.ipynb` and `deepsort_plugin.py` cannot be excuted by itself -- they requires input values that came out from `yolov4_plugin`.

## PyWaggle
- We use the two methods (`YOLOv4` and `DeepSort`) to calculate traffic state. <br/>
- And the calculation results are sent through `PyWaggle` to Cloud (for detail information see [PyWaggle](https://github.com/waggle-sensor/pywaggle)). <br/>
- The `PyWaggle` is simply implemented in `pywaggle.ipynb` and `pywaggle.py`, and still testing the modules (3/27/2021).


## Notes to developer:
- Timestamp for each traffic state needs to be the time when the video captured: The timestamp must be provided with the video.
- The waggle.plugin.publish function is tested, and checked with [log](https://github.com/waggle-sensor/pywaggle/wiki/Plugins:-Getting-Started#debug-logging).
- Not yet plugin-ized (3/30/2021): needs to be dockerized and create sage.json and others using [virtual waggle](https://github.com/waggle-sensor/virtual-waggle#running-node-application-stack)
