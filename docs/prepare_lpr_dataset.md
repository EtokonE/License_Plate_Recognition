#### Получение данных для обучения LPRNet из разметки детекции

```bash
$ python -m src.data.coco2lpr

ptoinal arguments:
 --image_dir                       Directory with original images
 --coco_json_file                  Json file with annotations in coco format
 --out_dir                         Directory to save results
```

**Итоговая структура директорий:**
```tree
├──train
    ├── ann
        ├──A645BH199.json
        ├──B857HP76.json
        ├──...
    ├── img
        ├──A645BH199.png
        ├──B857HP76.png
        ├──...
├──val
    ├──...
├──test
    ├──...
```
**JSON файлы разметки имеют следующую структуру:**

**`A393AT797.json`**
```json
{ 
  "description": "A393AT797", -- автомобильный номер
  "name": "A393AT797_13", -- название изображение с вырезанным номером
  "size": {"width": 118, "height": 31}, -- размеры вырезанного изображения
  "original_image": "09_50_38_8000000_18.png" -- оригинальное полноразмерное изображение
}
```