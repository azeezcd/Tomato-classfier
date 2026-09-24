from predict import get_prediction

result = get_prediction("test/bacterial.JPG")

for item in result:
    print(
        f"{item['class']:<55}"
        f"{item['confidence']:.2f}%"
    )