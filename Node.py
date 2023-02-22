class Node():
    def __init__(self, id_in_dataset, numerical_attributes, nominal_attributes, numerical_attributes_range=None, x=None, y=None):
        self.id_in_dataset = id_in_dataset
        self.cluster = None

        self.numerical_attributes = numerical_attributes
        self.numerical_attributes_range = numerical_attributes_range
        self.nominal_attributes = nominal_attributes

        # only for 2d, numerical data
        self.x = x
        self.y = y