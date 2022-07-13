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
>[{\"xmin\":1201.1260986328,\"ymin\":617.4149169922,\"xmax\":1283.7456054688,\"ymax\":639.5236816406,\"confidence\":0.9210925698,\"class\":0,\"name\":\"Plate\",\"Number\":\"T719CO76\"},{\"xmin\":503.8233032227,\"ymin\":312.5590820312,\"xmax\":547.1025390625,\"ymax\":326.4127502441,\"confidence\":0.9146454334,\"class\":0,\"name\":\"Plate\",\"Number\":null},{\"xmin\":1716.9141845703,\"ymin\":634.1428222656,\"xmax\":1811.7192382812,\"ymax\":658.8519287109,\"confidence\":0.9138192534,\"class\":0,\"name\":\"Plate\",\"Number\":\"A854BY79\"},{\"xmin\":800.6448364258,\"ymin\":478.3475036621,\"xmax\":868.5396118164,\"ymax\":501.6263427734,\"confidence\":0.9069643617,\"class\":0,\"name\":\"Plate\",\"Number\":\"C433AP76\"},{\"xmin\":577.5911865234,\"ymin\":377.7813110352,\"xmax\":630.0615234375,\"ymax\":397.3248901367,\"confidence\":0.9018586874,\"class\":0,\"name\":\"Plate\",\"Number\":null},{\"xmin\":118.2302017212,\"ymin\":649.8344116211,\"xmax\":219.3068389893,\"ymax\":677.2192382812,\"confidence\":0.8966100812,\"class\":0,\"name\":\"Plate\",\"Number\":\"A544YH76\"},{\"xmin\":1022.6343994141,\"ymin\":229.9101257324,\"xmax\":1049.3635253906,\"ymax\":240.8238525391,\"confidence\":0.7937896848,\"class\":0,\"name\":\"Plate\",\"Number\":null}]
   
