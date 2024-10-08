import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


def create_dataset(data_path, output_path=None, contrast_normalization=False, whiten=False, data_augmentation=False):
    """
    Reads and optionally preprocesses the data.

    Arguments
    --------
    data_path: (String), the path to the file containing the data
    output_path: (String), the name of the file to save the preprocessed data to (optional)
    contrast_normalization: (boolean), flags whether or not to normalize the data (optional). Default (False)
    whiten: (boolean), flags whether or not to whiten the data (optional). Default (False)

    Returns
    ------
    train_ds: (TensorDataset, the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    # read the data and extract the various sets

    # apply the necessary preprocessing as described in the assignment handout.
    # You must zero-center both the training and test data
    if data_path == "image_categorization_dataset.pt":
        # do mean centering here
        # Initialize the datatset, the training data, and the testing data
        dataset = torch.load(data_path)
        data_tr = dataset['data_tr']
        data_te = dataset['data_te']
        sets_tr = dataset['sets_tr']
        label_tr = dataset['label_tr']
        
        # Take the mean and collapse the first dimension (the number of pictures)
        mean = torch.mean(data_tr[sets_tr == 1], dim=0)
        data_tr = data_tr - mean
        data_te = data_te - mean
            
        # %%% DO NOT EDIT BELOW %%%% #
        if contrast_normalization:
            image_std = torch.std(data_tr[sets_tr == 1], dim = 0, unbiased=True)
            image_std[image_std == 0] = 1
            data_tr = data_tr / image_std
            data_te = data_te / image_std
        if whiten:
            examples, rows, cols, channels = data_tr.size()
            data_tr = data_tr.reshape(examples, -1)
            W = torch.matmul(data_tr[sets_tr == 1].T, data_tr[sets_tr == 1]) / examples
            E, V = torch.linalg.eigh(W)
            E = E.real
            V = V.real

            en = torch.sqrt(torch.mean(E).squeeze())
            M = torch.diag(en / torch.max(torch.sqrt(E.squeeze()), torch.tensor([10.0])))

            data_tr = torch.matmul(data_tr.mm(V.T), M.mm(V))
            data_tr = data_tr.reshape(examples, rows, cols, channels)

            data_te = data_te.reshape(-1, rows * cols * channels)
            data_te = torch.matmul(data_te.mm(V.T), M.mm(V))
            data_te = data_te.reshape(-1, rows, cols, channels)
        
        # New augmentation follows,
        # Create the transformation, stack the new data into the training data
        # Label all of them as training data on sets_tr, duplicate the labeling to append
        if data_augmentation:
            # random_horizontal:
            transform = transforms.RandomHorizontalFlip()
            new_data = transform(data_tr)
            data_tr = torch.vstack((data_tr, new_data))
            new_sets_tr = torch.ones(new_data.size(dim=0))
            sets_tr = torch.cat((sets_tr, new_sets_tr))
            label_tr = torch.cat((label_tr, label_tr.clone()))
        
            # Creating an 70/30 split for the training and validation
            mask = torch.rand(sets_tr.size(dim=0), 1)  # uniformly distributed between 0 and 1
            mask = mask < 0.3   # 70% pixels "on"  
            mask = mask + 1 
            sets_tr = mask.flatten()
        
        preprocessed_data = {"data_tr": data_tr, "data_te": data_te, "sets_tr": sets_tr, "label_tr": label_tr}
        if output_path:
            torch.save(preprocessed_data, output_path)

    train_ds = TensorDataset(data_tr[sets_tr == 1], label_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], label_tr[sets_tr == 2])

    return train_ds, val_ds
