from ultralytics import SAM
import time

# Load a model
# t / s / b / l
model = SAM("sam2_b.pt")

# Display model information (optional)
model.info()

# Run inference
start_time = time.time()
model("examples/soccer/data/output.mp4", device="cuda")
print(f"Finished in {time.time() - start_time:.2f} seconds")



from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam2_b.pt", device="cuda")

