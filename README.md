# License Plate Recognition
## Docker

```bash
$ docker pull etokone/license_plate_recognition:latest
$ docker run --gpus all --rm -it -p 8081:8080 etokone/license_plate_recognition:latest
```

## Prediction
```bash
$ curl -X POST -F file=@example_image.png http://0.0.0.0:8081/predict
```
>"{\"0\":{\"xmin\":1598.5034179688,\"ymin\":399.6758422852,\"xmax\":1652.3876953125,\"ymax\":415.1717834473,\"confidence\":0.9346209168,\"class\":0,\"name\":\"Plate\",\"Number\":null},\"1\":{\"xmin\":591.9412841797,\"ymin\":470.9756469727,\"xmax\":657.7363891602,\"ymax\":489.1953735352,\"confidence\":0.9281808734,\"class\":0,\"name\":\"Plate\",\"Number\":\"H639TO76\"},\"2\":{\"xmin\":972.0346679688,\"ymin\":429.9967346191,\"xmax\":1040.8781738281,\"ymax\":448.0764770508,\"confidence\":0.924728334,\"class\":0,\"name\":\"Plate\",\"Number\":\"T515HP76\"},\"3\":{\"xmin\":1311.666015625,\"ymin\":435.9352416992,\"xmax\":1380.8090820312,\"ymax\":454.9781494141,\"confidence\":0.9096539617,\"class\":0,\"name\":\"Plate\",\"Number\":\"P391HAX71\"},\"4\":{\"xmin\":677.5120849609,\"ymin\":211.3262023926,\"xmax\":720.3065795898,\"ymax\":223.1401672363,\"confidence\":0.8973209262,\"class\":0,\"name\":\"Plate\",\"Number\":null},\"5\":{\"xmin\":171.7325134277,\"ymin\":277.5382080078,\"xmax\":214.2937316895,\"ymax\":291.0704345703,\"confidence\":0.8591089249,\"class\":0,\"name\":\"Plate\",\"Number\":null},\"6\":{\"xmin\":1901.8439941406,\"ymin\":416.9491882324,\"xmax\":1920.0,\"ymax\":433.6493225098,\"confidence\":0.8447455764,\"class\":0,\"name\":\"Plate\",\"Number\":null},\"7\":{\"xmin\":281.625793457,\"ymin\":169.704284668,\"xmax\":331.4038696289,\"ymax\":195.6185913086,\"confidence\":0.6291396022,\"class\":0,\"name\":\"Plate\",\"Number\":null}}"
   
