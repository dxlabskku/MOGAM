# MOGAM: A Multimodal Object-oriented Graph Attention Model for Depression Detection
We propose the Multimodal Object-oriented Graph Attention Model (MOGAM) for early detection of depression in social media (YouTube). Unlike existing approaches, MOGAM is designed to handle diverse data types, such as text, images, and videos, making it more scalable and versatile. We ensure authenticity by including vlogs only from users with clinical depression diagnoses and incorporate additional metadata. MOGAM achieves an accuracy of 0.871 and an F1-score of 0.888. Our validation with a benchmark dataset shows comparable results to prior studies (0.61 F1-score).


## Workflow of our study
![flow](https://github.com/dxlabskku/MOGAM/assets/117570065/57a7da4a-d256-42c4-8168-935182bf984e)
**Figure : From data collcetion to classification**


## Data
![data_example](https://github.com/dxlabskku/MOGAM/assets/117570065/79aa7d44-5d5f-44b1-b375-e1297eb8673c)
**Figure : Data Example (```YouTube``` Channel)**

Our datasets consist of three categories: daily, depression, and high-risk potential.

## Experiment
Within the ```model``` folder we uploaded classification models to detect depression posts. We compared three graph neural networks (GCN, GraphSAGE, and GAT) with MOGAM.
A portion of the experiment results are shown below.

|                                      | Model                | Label               | Accuracy |      Precision |         Recall |       F1-Score |
|--------------------------------------|----------------------|---------------------|---------:|---------------:|---------------:|---------------:|
|      1. Daily versus Depression      | GAT                  | Daily               |    0.826 |          0.798 |          0.854 |          0.825 |
|                                      |                      | Depression          |          |          0.855 |          0.799 |          0.826 |
|                                      | MOGAM with GAT       | Daily               |    0.871 |          0.850 |          0.845 |          0.847 |
|                                      |                      | Depression          |          | \textbf{0.887} |          0.890 | \textbf{0.888} |
| 2. Daily versus High-risk Depression | GAT                  | Daily               |    0.823 |          0.844 |          0.930 |          0.885 |
|                                      |                      | High-risk potential |          |          0.735 |          0.529 |          0.615 |
|                                      | MOGAM with GAT       | Daily               |    0.996 |          0.995 | \textbf{1.000} | \textbf{0.997} |
|                                      |                      | High-risk potential |          | \textbf{1.000} |          0.986 |          0.993 |
