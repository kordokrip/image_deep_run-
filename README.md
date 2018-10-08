# image_deep_run-
inception-v3 used deep-run image analysis

materials

- python (latest version)
- tensorflow (latest version)
- scikit-learn
- numpy
- matplotlib

1. Prepare n input images as input

2. Convert the image to an array of (n, 299, 299 3)

3. Convert to an array of (n, 2048) through various processes

3. Fully connected, sorted into m classes

4. Change sum of m classes to 1 bysoftmax

5. Find the class with the highest number of armax
