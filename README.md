# image_deep_run-
inception-v3 used deep-run image analysis


Conventional image retrieval is based on CBIR (content based image retrieval) based image retrieval. There is a lucene-based LIRE (Lucene Image REtrieval) as an open source, and it is also a plug-in version of it in elasticsearch.

If you are studying deep running, you will know CNN (convolutional neural network). It is the most widely used algorithm for machine learning related to image, and it is an algorithm that shows excellent performance especially in image classification. Typically, Google's inception (googlenet), ms resNet, mobilenet, VGG.

In this project, we will look for images based on CNN rather than classification.

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
