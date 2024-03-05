def data_split(text, split_ratio, type):
    """
    Split the data into training and validation sets
    """
    
    if type =='cut':
        # Split the data
            # Calculate the length of the training set
        train_len = int(len(text) * split_ratio)
        
        # Split the data
        train_data = text[:train_len]
        val_data = text[train_len:]
        
        return train_data, val_data
    
    if type == 'random':
        # Split the data
        assert False # not implemented yet
        #split into block sizes and pick blocks at random

    
    return train_data, val_data