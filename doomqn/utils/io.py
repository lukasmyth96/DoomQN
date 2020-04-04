import pickle


def pickle_save(filepath, an_object):

    with open(filepath, 'wb') as f:
        pickle.dump(an_object, f, protocol=4)


def pickle_load(filepath):

    with open(filepath, 'rb') as f:
        an_object = pickle.load(f, encoding='bytes')

    return an_object