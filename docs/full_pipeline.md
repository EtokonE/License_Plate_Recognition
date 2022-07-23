## Запуск полного цикла распознавания 
```bash
$ python main.py

Optoinal arguments:
 --yolo_weights         default: ./models/best.pt                                  Pretrained yolov5 weights
 --lprnet_weights       default: ./models/LPRNet_Ep_BEST_model.ckpt                Pretrained LPRNet weights
 --stn_weights          default: ./models/SpatialTransformer_Ep_BEST_model.ckpt    Pretrained STNet weights
 --image                default: ./models/SpatialTransformer_Ep_BEST_model.ckpt    Pretrained STNet weights
 --image                default: reports/figures/yolo/09_38_54_8000000_15.png      Path to image
 --out_dir              default: reports/                                          Path to directory where would save results
 --save_json            default: True                                              Save JSON with results
 --save_img             default: True                                              Save image with results
 --save_dataframe       default: True                                              Save dataframe with results
```

### Результаты
****
`reports/inference_results/09_03_39_8000000_5_results.png`

<img src="../reports/inference_results/09_03_39_8000000_5_results.png" width="900 px"/>

`reports/inference_results/09_03_39_8000000_5_results.json`
```json
{
  "0":{"xmin":1598.5455322266,"ymin":399.8723449707,"xmax":1652.2009277344,"ymax":415.0353088379,
       "confidence":0.931964159,"class":0,"name":"Plate","Number":null},
  "1":{"xmin":591.9268798828,"ymin":470.7789306641,"xmax":658.0405883789,"ymax":489.2675476074,
       "confidence":0.9258311391,"class":0,"name":"Plate","Number":"H639TO76"},
  "2":{"xmin":971.9020996094,"ymin":430.119354248,"xmax":1040.9666748047,"ymax":448.1368408203,
       "confidence":0.921592474,"class":0,"name":"Plate","Number":"T515HP76"},
  "3":{"xmin":1311.4468994141,"ymin":436.5375671387,"xmax":1380.4725341797,"ymax":454.7528076172,
       "confidence":0.9069831967,"class":0,"name":"Plate","Number":"P391HA71"},
  "4":{"xmin":677.9398193359,"ymin":211.6175994873,"xmax":719.9579467773,"ymax":223.0309753418,
       "confidence":0.891178906,"class":0,"name":"Plate","Number":null},
  "5":{"xmin":171.3097686768,"ymin":277.8567504883,"xmax":214.3333282471,"ymax":291.2663879395,
       "confidence":0.8507843614,"class":0,"name":"Plate","Number":null},
  "6":{"xmin":1902.1220703125,"ymin":417.3900146484,"xmax":1920.0,"ymax":433.4660339355,
       "confidence":0.840703547,"class":0,"name":"Plate","Number":null}}
```

`reports/inference_results/09_03_39_8000000_5_results.csv`

|     | xmin | ymin | xmax | ymax | confidence | class | name | Number |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 1598.5455322265625 | 399.8723449707031 | 1652.200927734375 | 415.0353088378906 | 0.9319641590118408 | 0   | Plate |     |
| 1   | 591.9268798828125 | 470.7789306640625 | 658.0405883789062 | 489.2675476074219 | 0.925831139087677 | 0   | Plate | H639TO76 |
| 2   | 971.902099609375 | 430.1193542480469 | 1040.9666748046875 | 448.1368408203125 | 0.9215924739837646 | 0   | Plate | T515HP76 |
| 3   | 1311.4468994140625 | 436.5375671386719 | 1380.4725341796875 | 454.7528076171875 | 0.9069831967353821 | 0   | Plate | P391HA71 |
| 4   | 677.9398193359375 | 211.6175994873047 | 719.9579467773438 | 223.03097534179688 | 0.8911789059638977 | 0   | Plate |     | 
| 5   | 171.3097686767578 | 277.85675048828125 | 214.3333282470703 | 291.2663879394531 | 0.8507843613624573 | 0   | Plate |     | 
| 6   | 1902.1220703125 | 417.3900146484375 | 1920.0 | 433.4660339355469 | 0.840703547000885 | 0   | Plate |     | 
