# Vietnamese License Plate Recognition
## Tasks
* Detect license plate(s) in an image, a video or realtime.
* Extract characters from detected license plate(s).

## How to inference
First of all, make sure that you have installed all requirements. Then, to inference, you can run the Python file with the following code

```
python detect.py \
        --modality <image-or-video-or-realtime> \
        --confidence <confidence-threshold-for-ocr> \
        --src <path-to-videos-or-images> \
        --out <path-to-output-directory>
```

Or the provided script files

```
# For image
bash detect_image.sh

# For video
bash detect_video.sh

# For realtime
bash detect_realtime.sh
```

## Results
![alt text](out/images/1.jpg)
![alt text](out/images/5.jpg)

## Challenges
Cannot detect or OCR on license plates in the environment that:
* Too bright
![alt text](out/images/2.jpg)
* Too blurry
![alt text](out/images/blur.jpg)
* License plates with characters that are not black (because the OCR model was not trained for these cases)
![alt text](out/images/8.jpg)
* License plates that are too small
![alt text](out/images/557.jpg)

## Refereces
https://github.com/trungdinh22/License-Plate-Recognition
https://github.com/WaseemKn/FuzzyColorImageEnhancement-FuzzyLogicCourse-ITE5thYear