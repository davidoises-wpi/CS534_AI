Object Detection Models Evaluation With EuroCity Person Dataset

Dataset Processing:

1. Follow the ECPB (Euro City Person Benchmark) instructions to download the required datasets
2. To get started go to the DetectionAlgorithm models and add any models to be evaluated
3. Modify DatasetDetect_process_all.py to call the models that will be used for evaluation
4. Process the dataset running the script from step 3

Models Evaluation:

1. Got to the DatasetResultsEvaluation folder
2. Modify DatasetEvaluation_process_all.py to use the desired result set as well as labeled dataset
3. Review results in terms of execution time and Log Average Miss Rate