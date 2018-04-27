import tensorflow as tf

def concat(*ds_elements):
    #Create one empty list for each component of the dataset
    lists = [[] for _ in ds_elements[0]]
    l = 0
    
    for element in ds_elements:
        for i, tensor in enumerate(element):
            #For each element, add all its component to the associated list                        
            if i==1:
                l= tf.maximum(l,tf.shape(tensor)[1])
    
    for element in ds_elements:
        for i, tensor in enumerate(element):
            #For each element, add all its component to the associated list                       
            if i==1:
                tensor_ = tf.pad(tensor,[[0,0],[0,l-tf.shape(tensor)[1]],[0,0]], mode="CONSTANT")                          
                lists[i].append(tensor_)
            else:
                lists[i].append(tensor)
    
    #Concatenate each component list
    return tuple((tf.reshape(tf.stack(lists[0]),[-1]), tf.concat(lists[1],0), tf.concat(lists[2],0)))

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                        'labels': tf.FixedLenFeature([], tf.int64),
                        'shapes': tf.FixedLenFeature([2], tf.int64),
                        'features': tf.VarLenFeature( tf.float32)
            })

    labels = features['labels']
    shapes = features['shapes']
    feats = features['features']
    #print(shapes)
    shapes = tf.cast(shapes, tf.int32)
    feats2d = tf.reshape(feats.values, shapes)
    #print(shapes)
    return labels, feats2d, shapes

def binary_decode_(class_):
    def binary_decode(serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                        'labels': tf.FixedLenFeature([], tf.int64),
                        'shapes': tf.FixedLenFeature([2], tf.int64),
                        'features': tf.VarLenFeature( tf.float32)
            })

        labels = tf.cast(tf.equal(features['labels'], class_), tf.int64)
        shapes = features['shapes']
        feats = features['features']
    #print(shapes)
        shapes = tf.cast(shapes, tf.int32)
        feats2d = tf.reshape(feats.values, shapes)
    #print(shapes)
        return labels, feats2d, shapes
    return binary_decode

# In[3]:


def get_padded_shapes(dataset):
    """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.
    Args:
    dataset: The dataset that will be batched with padding.
    Returns:
    The same structure as ``dataset.output_shapes`` containing the padded
    shapes.
    """
    return tf.contrib.framework.nest.map_structure(lambda shape: shape.as_list(), dataset.output_shapes)


# In[4]:


def get_dataset_shape(filename):
    c = 0
    for record in tf.python_io.tf_record_iterator(filename):
        c += 1
    return c

def inputs(filename, batch_size, shuffle=True):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
    Returns:
        A tuple (feats2d, labels).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)
        # Would be better if data was shuffled but make the prgoram crash
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        #print(dataset)
        dataset = dataset.repeat()

        #dataset = dataset.batch(batch_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes=get_padded_shapes(dataset))
        iterator = dataset.make_one_shot_iterator()
        #iterator = dataset.make_initializable_iterator()

    return iterator.get_next()

def balanced_inputs(filename, batch_size, num_class, shuffle=True):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
    Returns:
        A tuple (feats2d, labels).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)
        # Would be better if data was shuffled but make the prgoram crash
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        #print(dataset)
        dataset = dataset.repeat()
        dataset_per_class = [None]*num_class
        for i in range(num_class):
            dataset_per_class[i] = dataset.filter(lambda lb,sh,ft: tf.equal(lb,i))
            dataset_per_class[i] = dataset_per_class[i].padded_batch(int(batch_size/num_class), padded_shapes=get_padded_shapes(dataset_per_class[i]))
        zipped_ds = tf.data.Dataset.zip(tuple(dataset_per_class))
        dataset = zipped_ds.map(concat)
        #dataset = dataset.padded_batch(int(batch_size/num_class), padded_shapes=get_padded_shapes(dataset))        
        iterator = dataset.make_one_shot_iterator()
        #iterator = dataset.make_initializable_iterator()

    return iterator.get_next()


def binary_inputs(filename, batch_size, class_, shuffle=True):
    """
    Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
    Returns:
        A tuple (feats2d, labels).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    
    """

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(binary_decode_(class_))
        # Would be better if data was shuffled but make the prgoram crash
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        #print(dataset)
        dataset = dataset.repeat()

        #dataset = dataset.batch(batch_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes=get_padded_shapes(dataset))
        iterator = dataset.make_one_shot_iterator()
        #iterator = dataset.make_initializable_iterator()

    return iterator.get_next()

def balanced_binary_inputs(filename, batch_size, class_, shuffle=True):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
    Returns:
        A tuple (feats2d, labels).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(binary_decode_(class_))
        # Would be better if data was shuffled but make the prgoram crash
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        #print(dataset)
        dataset = dataset.repeat()
        dataset_per_class = [None]* 2
        for i in range(2):
            dataset_per_class[i] = dataset.filter(lambda lb,sh,ft: tf.equal(lb,i))
            dataset_per_class[i] = dataset_per_class[i].padded_batch(int(batch_size/2), padded_shapes=get_padded_shapes(dataset_per_class[i]))
        zipped_ds = tf.data.Dataset.zip(tuple(dataset_per_class))
        dataset = zipped_ds.map(concat)
        #dataset = dataset.batch(batch_size)
        #dataset = dataset.padded_batch(batch_size, padded_shapes=get_padded_shapes(dataset))
        iterator = dataset.make_one_shot_iterator()
        #iterator = dataset.make_initializable_iterator()

    return iterator.get_next()
