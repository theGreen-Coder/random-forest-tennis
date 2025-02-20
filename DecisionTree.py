import numpy as np

class TreeNode:
    def __init__(self, val=None, feature=None, threshold=None, left=None, right=None):
        self.val = val
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = threshold
        self.data = None
    
    def calculate_gini(self, data):
        true_val = data[data[:, data.shape[1]-1] == 1]
        false_val = data[data[:, data.shape[1]-1] == 0]

        return 1 - (len(true_val)/len(data))**2 - (len(false_val)/len(data))**2
    
    # Requires last column to be what we're trying to predict
    def fit(self, data, depth, max_depth):
        count_true = len(data[data[:, data.shape[1]-1] == 1])
        count_false = len(data[data[:, data.shape[1]-1] == 0])

        if depth >= max_depth or count_true == 0 or count_false == 0:
            if count_true >= count_false:
                self.val = 1
            else:
                self.val = 0
            
            return self

        self.data = data

        lowest_gini = float("inf")
        lowest_gini_feature = -1
        lowest_gini_threshold = -1

        for i in range(0, data.shape[1] - 1):
            # print(i)
            # print(np.unique(data[:, i]))

            unique_values = np.unique(data[:, i])

            # Calculate gini impurity
            for j in range(unique_values.shape[0]-1):
                filter_value = (unique_values[j]+unique_values[j+1])/2
                # print(filter_value)
            
                left = data[data[:, i] <= filter_value]
                right = data[data[:, i] > filter_value]
                # print(left)
                # print(right)
                assert len(left) + len(right) == len(data)

                total_gini_impurity = (len(left)/len(data))*self.calculate_gini(left) + (len(right)/len(data))*self.calculate_gini(right)
                # print(total_gini_impurity)

                if total_gini_impurity < lowest_gini:
                    lowest_gini = total_gini_impurity
                    lowest_gini_feature = i
                    lowest_gini_threshold = filter_value
        
        left = data[data[:, lowest_gini_feature] < lowest_gini_threshold]
        right = data[data[:, lowest_gini_feature] >= lowest_gini_threshold]
        
        self.feature = lowest_gini_feature
        self.threshold = lowest_gini_threshold

        self.left = TreeNode().fit(data=left, depth=depth+1, max_depth=max_depth)
        self.right = TreeNode().fit(data=right, depth=depth+1, max_depth=max_depth)

        return self

class DecisionTree:
    def __init__(self, tree=None, max_depth=float("inf"), min_samples_split=2, min_samples_leaf=1):
        self.tree = tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def build_tree(self, data):
        if self.tree is not None:
            raise Exception("Error in building decision tree. Tree is not None.")
        
        else:
            head = TreeNode()
            head = head.fit(data=data, depth=0, max_depth=self.max_depth)

            self.tree = head
    
    def print_tree(self):
        def print_recursive(node, depth=0):
            if node is None:
                return
            
            indent = "  " * depth  # Indentation for better visualization
            
            if node.val is not None:  # Leaf node
                print(f"{indent}Leaf: {node.val}")
            
            else:  # Decision node
                print(f"{indent}Feature {node.feature} < {node.threshold}")
                print(f"{indent}Left:")
                print_recursive(node.left, depth + 1)
                print(f"{indent}Right:")
                print_recursive(node.right, depth + 1)

        print_recursive(self.tree)
