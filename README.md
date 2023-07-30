# MOGAM: A Multimodal Object-oriented Graph Attention Model for Depression Detection
We propose the Multimodal Object-oriented Graph Attention Model (MOGAM) for early detection of depression in social media (YouTube). Unlike existing approaches, MOGAM is designed to handle diverse data types, such as text, images, and videos, making it more scalable and versatile. We ensure authenticity by including vlogs only from users with clinical depression diagnoses and incorporate additional metadata. MOGAM achieves an accuracy of 0.871 and an F1-score of 0.888. Our validation with a benchmark dataset shows comparable results to prior studies (0.61 F1-score).


## Workflow of our study
![flow](https://github.com/dxlabskku/MOGAM/assets/117570065/57a7da4a-d256-42c4-8168-935182bf984e)
**Figure : From data collcetion to classification**


## Data
![data_example](https://github.com/dxlabskku/MOGAM/assets/117570065/79aa7d44-5d5f-44b1-b375-e1297eb8673c)
**Figure : Data Example (```YouTube``` Channel)**

Our datasets consist of three categories: daily, depression, and high-risk potential. 
| Dataset             | Vlogs | Average Duration |
|---------------------|-------|------------------|
| Daily               | 1888  | 903.39s          |
| Depression          | 2237  | 416.03s          |
| High-risk potential | 642   | 515.74s          |

## Experiment
Within the ```model``` folder we uploaded classification models to detect depression posts. We compared three graph neural networks (GCN, GraphSAGE, and GAT) with MOGAM. Overall, MOGAM improved the performance of the baseline models, expecially, MOGAM with GAT based model achieved the highest F1-score (0.888) among the baselines, indicating that the cross-attnetion mechanism and multimodal featuers are suitbale approach for detecting depression symptoms and patters in vlogs.
