from MLManager import MLManager

m = MLManager("best.pt")
val1 = "batch_1"
val2 = "images-to-infer/000104.JPG"
print(m.predict_image(val2))