from ultralytics import SAM
import time
#
# # Load a model
# # t / s / b / l
model = SAM("sam2_t.pt")

# Display model information (optional)
model.info()

# Run inference
start_time = time.time()
model("examples/soccer/data/input.mp4", device="cuda", save=True)
print(f"Finished in {time.time() - start_time:.2f} seconds")

#
#
# from ultralytics.data.annotator import auto_annotate
#
# auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam2_b.pt", device="cuda")
#

# from ultralytics import YOLO
#
# # Create a YOLO-World model
# model = YOLO("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
# model.set_classes(["red vest"])
#
# # Track with a YOLO-World model on a video
# results = model.predict(source="examples/soccer/data/short_output.mp4", save=True, device="mps", conf=0.1)
# # save it as video