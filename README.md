# MOGAM: A Multimodal Object-oriented Graph Attention Model for Depression Detection
We propose the Multimodal Object-oriented Graph Attention Model (MOGAM) for early detection of depression in social media (YouTube). Unlike existing approaches, MOGAM is designed to handle diverse data types, such as text, images, and videos, making it more scalable and versatile. We ensure authenticity by including vlogs only from users with clinical depression diagnoses and incorporate additional metadata. MOGAM achieves an accuracy of 0.871 and an F1-score of 0.888. Our validation with a benchmark dataset shows comparable results to prior studies (0.61 F1-score).


## Workflow of our study
![flow](/image/flow.jpg)
**Figure : From data collcetion to classification**


## Data
We collected vlogs from YouTube and with two hashtags: #일상브이로그 and #우울증브이로그 that represent #daily vlog and #depression vlog. Then we curated depression vlogs with clinical diagnosis to guarantee the vlogs' relevance to a medical depression diagnosis. Based on the upload time of each depression diagnosis vlog, we divided the user’s vlog list into two groups: high-risk potential vlogs, which were uploaded before depression diagnosis vlog, and depression vlogs, which were uploaded after depression diagnosis vlog. The final datasets consist of three categories: daily, depression, and high-risk potential. 

| Dataset             | Vlogs | Average Duration |
|---------------------|-------|------------------|
| Daily               | 1888  | 903.39s          |
| Depression          | 2237  | 416.03s          |
| High-risk potential | 642   | 515.74s          |

![data_example](/image/example.jpg)
**Figure : Data Example (```YouTube``` Channel)**

## Experiment
Within the ```model``` folder we uploaded classification models to detect depression posts. We compared three graph neural networks (GCN, GraphSAGE, and GAT) with MOGAM. Overall, MOGAM improved the performance of the baseline models, expecially, MOGAM with GAT based model achieved the highest F1-score (0.888) among the baselines, indicating that the cross-attnetion mechanism and multimodal featuers are suitbale approach for detecting depression symptoms and patters in vlogs.
