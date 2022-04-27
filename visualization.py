from utils import *
import numpy as np

def main(output_path, datasets):
    methods = ['t-SNE', 'UMAP', 'PaCMAP', 'h-NNE']

    args = {
        'UMAP': [{'n_neighbors': 10}, {'n_neighbors': 20}, {'n_neighbors': 40}],
        'PaCMAP': [{'n_neighbors': 10}, {'n_neighbors': 20}, {'n_neighbors': 40}],
        't-SNE': [{'perplexity': 10}, {'perplexity': 20}, {'perplexity': 40}],
        'h-NNE': [{'radius': 0.20}, {'radius': 0.45}, {'radius': 0.60}],
    }

    print("Experiment started")
    for dataset_name in datasets:
        embeddings = []
        labelset = []
        titles = []
        for method in methods:
            parameters = args[method]
            for parameter in parameters:
                param1 = list(parameter.keys())[0]
                param2 = parameter[param1]
                cur_name = '{dataset_name}_{method}_{param1}_{param2}'.format(dataset_name=dataset_name, method=method,
                                                                              param1=param1, param2=param2)
                cur_name = output_path + cur_name
                X_low = np.load(cur_name + "-low_features" + ".npy")
                labels = np.load(cur_name + "-labels" + ".npy")
                embeddings.append(X_low)
                labelset.append(labels)
                titles.append(u'{}\n{}={}'.format(method, param1, param2))
        generate_combined_figure(embeddings, labelset, titles, dataset_name)
    print('Finished')
    return 0

if __name__ == '__main__':
    # Please define the output_path here
    output_path = "./output/"
    datasets = ['MNIST', 'CIFAR-10', 'SVHN']
    main(output_path, datasets)