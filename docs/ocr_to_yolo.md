#### Формат аннотация датасета OCR для конкретного фрэйма:
**`frame000060.txt`**
```text
    vehicles: 2
position_vehicle: 768 122 555 478
	type: car
plate: ARG4068
position_plate: 1115 551 93 34
	char 1: 1123 564 9 17
	char 2: 1133 564 11 18
	char 3: 1144 564 10 16
	char 4: 1160 562 9 16
	char 5: 1170 562 10 16
	char 6: 1181 562 10 16
	char 7: 1191 561 10 16

position_vehicle: 1403 0 348 190
	type: car
plate: AZI4586
position_plate: 1582 127 65 25
	char 1: 1587 133 7 12
	char 2: 1594 134 9 12
	char 3: 1603 134 5 13
	char 4: 1613 136 7 12
	    char 5: 1620 136 9 13
	char 6: 1628 137 8 12
	char 7: 1636 138 7 11

```    
#### Преобразовать в YOLO формат:
```bash
$ python -m src.data.ocr2yolo.py 

Optoinal arguments:
--ocr_folder                      default=PATH             path to ocr data
```
**В результате работы скрипта получаем следующую разметку для каждого кадра:**

**`frame000060.txt`**
```text
0 1115 551 93 34
0 1582 127 65 25
```
**`train_ocs.txt`**
```text
data/obj_train_data/ocr/frame000060.txt
```

#### Если в дальнейшем необходима нормализация координат в разметке YOLO используем следующий скрипт
````bash
$ python -m src.data.yolo2yolo_norm

Optoinal arguments:
 --yolo_folder                     default: PATH,          path to the folder containing the labelling in yolo format
 --img_height                      default: 1080,          height of images
 --img_width                       default: 1920,          width of images  
````
