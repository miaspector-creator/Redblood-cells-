---
title: Blood Cell Counter
emoji: 🩸
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: AI blood smear cell counter and WBC differential (research)
---

# Blood Cell Counter

Counts red blood cells, white blood cells and platelets in blood smear images, types each white
blood cell (neutrophil, lymphocyte, monocyte, eosinophil, basophil, immature granulocyte), and runs a
guided white cell differential from a live microscope camera.

- Detector: YOLO26s fine-tuned on TXL-PBC
- White cell classifier: YOLO26s-cls fine-tuned on BloodMNIST
- Source: https://github.com/miaspector-creator/Redblood-cells-

> **Disclaimer:** this is a master's research project for education and research only. It is not a
> medical device. AI can make mistakes, so counts and cell types may be wrong, and nothing it produces
> is a diagnosis. Always consult a qualified healthcare provider or a certified laboratory for medical
> advice and an accurate diagnosis.

This Space runs on CPU, so live camera mode analyzes fewer frames per second than on a computer with
a GPU. Images are processed in memory and are not stored.
