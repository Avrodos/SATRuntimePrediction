# SATRuntimePrediction

This is my master thesis project. For details, please look into the thesis PDF. If you have any questions, feel free to send me a message.

## Abstract:

The Boolean satisfiability problem (SAT) is a classic NP-complete problem, whose instances can take anywhere from a fraction of a second to potentially many years to solve. While there are some contributions in the field of runtime and hardness prediction of SAT instances, these mostly focus on single solvers with few instance families. Moreover, these works neglect the time complexity of the pipeline. This leads to limited generalizability and usability. 

In this work, we present a new practical set of features for predicting instance hardness and runtime. We also present ways to minimize the time complexity. In this context, we linked, among other contributions, the number of connected components to the instance family and the instance runtime.

To ensure high generalizability, we trained our models on a dataset with a very large number of instance families and with labels created from the runtimes of many different solvers. The results were highly competitive and scalable. We verified them using the established methods of k-fold cross-validation and hold-out sets. 

## Technology Stack: 
All runtime data was collected on a server-cluster, where each node is equipped with 2 Intel Xeon E5430 CPUs running at 2.66 GHz and 32 GB of RAM. All heavy computing, such as the graph generation or feature extraction, was executed in parallel on a server-cluster, where each node is equipped with 4 Intel Xeon E5-4640 CPUs running at 2.4 GHz with 512 GiB ECC RAM.

All processing steps were implemented as Python scripts, using Python 3.10, if not mentioned otherwise. \
For feature extraction, we used the NetworKit and NetworkX libraries. They are open-source libraries that provide flexible data structures (e.g. graphs), and a wide range of highly efficient algorithms for analysing networks. \
For efficient and fast parallel computing, we implemented all data structures and calculations using the Pandas and NumPy libraries. \
For visualization, e.g. plot generation, we used the established Matplotlib library. \
Finally, for our models we used the Scikit-learn library, as it is a popular open source machine learning library built on NumPy, SciPy and Matplotlib.

