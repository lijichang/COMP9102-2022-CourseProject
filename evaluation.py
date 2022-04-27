import os.path
import pickle
import numpy as np
from time import time
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10, SVHN, MNIST
from sklearn.manifold import TSNE
import umap
from pacmap import PaCMAP
from hnne import HNNE
from utils import evaluate_output

def data_prep(data_path, dataset='MNIST', size=10000):
    '''
    This function loads the dataset as numpy array.
    Input:
        data_path: path of the folder you store all the data needed.
        dataset: the name of the dataset.
        size: the size of the dataset. This is useful when you only
              want to pick a subset of the data
    Output:
        X: the dataset in numpy array
        labels: the labels of the dataset.
    '''
    if dataset == 'MNIST':
        mnist = MNIST(root=data_path, download=True, train=True)
        X = mnist.train_data.reshape((len(mnist.train_data), -1))
        labels = mnist.train_labels.reshape((len(mnist.train_data), -1))
        X = X.numpy()
        labels = labels.numpy()
    elif dataset == 'CIFAR-10':
        cifar10 = CIFAR10(root=data_path, download=True, train=True)
        X = cifar10.train_data.reshape((len(cifar10.train_data), -1))
        labels = cifar10.train_labels
        labels = np.array(labels).reshape((len(cifar10.train_data), -1))
    elif dataset == 'SVHN':
        svhn = SVHN(root=data_path, download=True, split='train')
        X = svhn.data.reshape((len(svhn.data), -1))
        labels = svhn.labels
        labels = np.array(labels).reshape((len(svhn.data), -1))
    else:
        print('Unsupported dataset')
        assert(False)
    return X[:size], labels[:size]

def experiment(X, method='t-SNE', **kwargs):
    if method == 't-SNE':
        transformer = TSNE(**kwargs)
    elif method == 'UMAP':
        transformer = umap.UMAP(**kwargs)
    elif method == 'PaCMAP':
        transformer = PaCMAP(**kwargs)
    elif method == 'h-NNE':
        transformer = HNNE(**kwargs)
    else:
        print("Incorrect method specified")
        assert(False)
    start_time = time()
    X_low = transformer.fit_transform(X)
    total_time = time() - start_time
    print("This run's time:")
    print(total_time)
    return X_low, total_time


def main(data_path, output_path, dataset_name='MNIST', csv_log='record.csv', size=10000000):
    print("dataset_name", dataset_name)
    X, labels = data_prep(data_path, dataset=dataset_name, size=size) #size
    if X.shape[1] > 100:
        pca = PCA(n_components=100)
        X = pca.fit_transform(X)
    print("Data loaded successfully")

    # do experiment
    methods = ['t-SNE', 'UMAP', 'PaCMAP', 'h-NNE']
    args = {
        't-SNE': [{'perplexity': 10}, {'perplexity': 20}, {'perplexity': 40}],
        'UMAP': [{'n_neighbors': 10}, {'n_neighbors': 20}, {'n_neighbors': 40}],
        'PaCMAP': [{'n_neighbors': 10}, {'n_neighbors': 20}, {'n_neighbors': 40}],
        'h-NNE': [{'radius': 0.20}, {'radius': 0.45}, {'radius': 0.60}],
    }

    print("Experiment started")
    all_results = {}
    for method in methods:
        parameters = args[method]
        for parameter in parameters:
            X_low, total_time = experiment(X, method, **parameter)
            param1 = list(parameter.keys())[0]
            param2 = parameter[param1]
            cur_name = '{dataset_name}_{method}_{param1}_{param2}'.format(dataset_name=dataset_name, method=method, param1=param1, param2=param2)
            X_low = X_low.reshape((X.shape[0], -1))
            results = evaluate_output(X, X_low, y=labels, name=cur_name)
            results['time'] = total_time
            cur_name = output_path + cur_name
            all_results[cur_name] = results
            np.save(cur_name+"-original_features", X)
            np.save(cur_name+"-low_features", X_low)
            np.save(cur_name+"-labels", labels)
            csv_print =  dataset_name + ', ' + method + ', ' + str(param1) + ', ' + str(param2) + ', ' +\
                            str(results['baseline_knn'][0]) + ', ' +str(results['baseline_knn'][1]) + ', ' +str(results['baseline_knn'][2]) + ', ' +str(results['baseline_knn'][3]) + ', ' +\
                                str(results['baseline_knn'][4]) + ', ' +str(results['baseline_knn'][5]) + ', ' +str(results['baseline_knn'][6]) + ', ' +str(results['baseline_knn'][7]) + ', ' +\
                                    str(results['baseline_svm']) + ', ' +\
                                        str(results['knn'][0]) + ', ' +str(results['knn'][1]) + ', ' +str(results['knn'][2]) + ', ' +str(results['knn'][3]) + ', ' +\
                                            str(results['knn'][4]) + ', ' +str(results['knn'][5]) + ', ' +str(results['knn'][6]) + ', ' +str(results['knn'][7]) + ', ' +\
                                                str(results['svm']) + ', ' + str(results['cte']) + ', ' + str(results['rte']) + ', ' + str(results['time'])
            csv_log.write(csv_print+"\r")
            csv_log.flush()
    with open(dataset_name, 'wb') as fp:
        pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finished')

    return 0

if __name__ == '__main__':
    # Please define the data_path and output_path here
    data_path = "./data/"
    output_path = "./output/"
    csv_log = open(os.path.join(output_path, 'record.csv'),'w')

    csv_print =  'Dataset' + ', ' + 'Method' + ', ' + 'Parameter' + ', ' + 'Value' + ', ' +\
                    'baseline_knn(nn=1)' + ', ' +'baseline_knn(nn=3)' + ', ' +'baseline_knn(nn=5)' + ', ' +'baseline_knn(nn=10)' + ', ' +\
                        'baseline_knn(nn=15)' + ', ' +'baseline_knn(nn=20)' + ', ' +'baseline_knn(nn=25)' + ', ' +'baseline_knn(nn=30)' + ', ' +\
                            'baseline_svm' + ', ' +\
                                'knn(nn=1)' + ', ' +'knn(nn=3)' + ', ' +'knn(nn=5)' + ', ' +'knn(nn=10)' + ', ' +\
                                    'knn(nn=15)' + ', ' +'knn(nn=20)' + ', ' +'knn(nn=25)' + ', ' +'knn(nn=30)' + ', ' +\
                                        'SVM Accuracy' + ', ' + 'Centroid Triplet Accuracy' + ', ' + 'Random Triplet Accuracy' + ', ' + 'Time'
    csv_log.write(csv_print+"\r")
    csv_log.flush()

    main(data_path, output_path, 'MNIST', csv_log)
    main(data_path, output_path, 'CIFAR-10', csv_log, 10000000)
    main(data_path, output_path, 'SVHN', csv_log, 10000000)

    csv_log.close()
