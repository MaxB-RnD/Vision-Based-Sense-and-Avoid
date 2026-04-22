from transformers import RTDetrConfig, RTDetrForObjectDetection

# 1. Define your labels exactly as they were for your "old" weights
id2label = {0: "drone", 1: "Aeroplane", 2: "Bird", 3: "Drone", 4: "Helicopter"}
label2id = {v: k for k, v in id2label.items()}

# 2. Load the base configuration for your specific architecture (v2 r50)
# This ensures all the technical "brain" settings are correct
config = RTDetrConfig.from_pretrained(
    "PekingU/rtdetr_v2_r50vd", 
    id2label=id2label, 
    label2id=label2id
)

# 3. Save it to your folder
config.save_pretrained("./configs/models/baselines/transformer_detector")

print("Success! A valid config.json has been created.")