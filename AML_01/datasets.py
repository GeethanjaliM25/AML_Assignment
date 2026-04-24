"""
datasets.py - Create and manage datasets for concept learning
"""

class EnjoySportDataset:
    """
    The classic EnjoySport dataset from Tom Mitchell's Machine Learning
    Target concept: "Days on which Aldo enjoys playing tennis"
    """
    
    def __init__(self):
        self.attributes = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']
        self.attribute_values = {
            'Sky': ['Sunny', 'Cloudy', 'Rainy'],
            'Temp': ['Warm', 'Cold'],
            'Humidity': ['Normal', 'High'],
            'Wind': ['Strong', 'Weak'],
            'Water': ['Warm', 'Cool'],
            'Forecast': ['Same', 'Change']
        }
        
        # Training examples
        self.X = [
            ('Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'),    # Example 1
            ('Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'),      # Example 2
            ('Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'),    # Example 3
            ('Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change')     # Example 4
        ]
        
        self.y = [1, 1, 0, 1]  # 1 = Enjoy, 0 = Not Enjoy
    
    def get_data(self):
        """Return training data"""
        return self.X, self.y
    
    def get_attributes(self):
        """Return attribute names"""
        return self.attributes
    
    def get_attribute_values(self):
        """Return possible values for each attribute"""
        return self.attribute_values
    
    def get_description(self):
        """Return dataset description"""
        return {
            'name': 'EnjoySport',
            'source': 'Tom Mitchell, Machine Learning (1997)',
            'instances': len(self.X),
            'attributes': self.attributes,
            'positive_count': sum(self.y),
            'negative_count': len(self.y) - sum(self.y)
        }


class CustomDataset:
    """
    Create custom dataset for testing edge cases
    """
    
    @staticmethod
    def create_xor_dataset():
        """
        XOR pattern - impossible for conjunctive learning
        Sky=Sunny AND Temp=Warm -> Enjoy
        Sky=Rainy AND Temp=Cold -> Enjoy
        Others -> Not Enjoy
        """
        attributes = ['Sky', 'Temp']
        X = [
            ('Sunny', 'Warm'),  # Enjoy
            ('Rainy', 'Cold'),  # Enjoy
            ('Sunny', 'Cold'),  # Not Enjoy
            ('Rainy', 'Warm')   # Not Enjoy
        ]
        y = [1, 1, 0, 0]
        return attributes, X, y
    
    @staticmethod
    def create_inconsistent_dataset():
        """
        Same instance with different labels - inconsistent data
        """
        attributes = ['Sky', 'Temp']
        X = [
            ('Sunny', 'Warm'),  # Enjoy
            ('Sunny', 'Warm'),  # Not Enjoy (contradiction)
            ('Rainy', 'Cold')   # Not Enjoy
        ]
        y = [1, 0, 0]
        return attributes, X, y
    
    @staticmethod
    def create_no_positive_dataset():
        """
        All negative examples - no positive to start Find-S
        """
        attributes = ['Sky', 'Temp']
        X = [
            ('Sunny', 'Cold'),
            ('Rainy', 'Warm'),
            ('Cloudy', 'Cold')
        ]
        y = [0, 0, 0]
        return attributes, X, y