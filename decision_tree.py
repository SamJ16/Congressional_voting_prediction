import sys
import csv
import math
import random

class DecisionNode:
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_name, test_index):
        self.test_name = test_name  # the name of the attribute to test at this node
        self.test_index = test_index  # the index of the attribute to test at this node

        self.children = {}  # dictionary mapping values of the test attribute to subtrees,
                            # where each subtree is either a DecisionNode or a LeafNode

    def classify(self, example):
        """Classify an example based on its test attribute value."""
        # print(example, type(example), self.test_index)
        test_val = example[self.test_index]
        # print(test_val)
        assert(test_val in self.children)
        return self.children[test_val].classify(example)

    def add_child(self, val, subtree):
        """Add a child node, which could be either a DecisionNode or a LeafNode."""
        self.children[val] = subtree

    def to_str(self, level=0):
        """Return a string representation of this (sub)tree."""
        # print(self.children)
        prefix = "\t"*(level+1)
        s = prefix + "test: " + self.test_name + "\n"
        for val, subtree in sorted(self.children.items()):
            s += "{}\t{}={} ->\n".format(prefix, self.test_name, val)
            s += subtree.to_str(level+1)
        return s


class LeafNode:
    """A leaf holds only a predicted class, with no test."""

    def __init__(self, pred_class, prob):
        self.pred_class = pred_class
        self.prob = prob

    def classify(self, example):
        # print(example)
        return self.pred_class, self.prob

    def to_str(self, level):
        """Return a string representation of this leaf."""
        prefix = "\t"*(level+1)
        return "{}predicted class: {} ({})".format(prefix, self.pred_class, self.prob)


class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, csv_path):
        """The constructor reads in data from a csv containing a header with feature names."""
        with open(csv_path, 'r') as infile:
            csvreader = csv.reader(infile)
            self.feature_names = next(csvreader)
            self.data = [row for row in csvreader]
            self.domains = [list(set(x)) for x in zip(*self.data)]
        self.root = None
    
    def entropy(self, attribute, target_name):
        attr_ind = self.feature_names.index(attribute)
        targ_ind = self.feature_names.index(target_name)
        uniq_attrs = set([row[attr_ind] for row in self.data])
        uniq_targs = set([row[targ_ind] for row in self.data])
        pairs = {}
        for k1 in uniq_attrs:
            for k2 in uniq_targs:
                # print(k1, k2)
                if (k1, k2) not in [(row[attr_ind], row[targ_ind]) for row in self.data]:
                    pairs[k1+','+k2] = 0
                else:
                    for row in self.data:
                        if row[attr_ind]+','+row[targ_ind] not in pairs.keys():
                            pairs[row[attr_ind]+','+row[targ_ind]] = 1
                        elif row[attr_ind]+','+row[targ_ind] in pairs.keys():
                            pairs[row[attr_ind]+','+row[targ_ind]] += 1
        # print(pairs)
        cond_probs = 0.0
        for t in uniq_targs:
            for a in uniq_attrs:
                if pairs[a+','+t] == 0:
                    cond_probs += 0
                else:
                    Pa = sum([val for key, val in pairs.items() if a in key]) #sum of all values where the attribute has this particular value a or b or whatever
                    cond_probs += -1.0*float(pairs[a+','+t]/Pa)*math.log(float(pairs[a+','+t]/Pa),2)*float(Pa/len(self.data))
        return cond_probs

    def information_gain(self, attribute, target_name):
        Ea = self.entropy(attribute, target_name)
        uniq_targ_vals = set([row[self.feature_names.index(target_name)] for row in self.data])
        E = 0.0
        for val in uniq_targ_vals:
            val_count = 0
            for row in self.data:
                if row[self.feature_names.index(target_name)] == val:
                    val_count += 1
            E += -1.0*float(val_count/len(self.data))*math.log(float(val_count/len(self.data)),2)
        return E - Ea

    def learn(self, target_name, min_examples=0):
        """Build the decision tree based on entropy and information gain.
        Args:
            target_name: the name of the class label attribute
            min_examples: the minimum number of examples allowed in any leaf node
        """
        #Resetting default min_examples value to optimize to training set
        if min_examples == len(self.data) or min_examples == 0:
            pairs = {}
            for attribute in self.feature_names:
                if attribute is not target_name:
                    attr_ind = self.feature_names.index(attribute)
                    targ_ind = self.feature_names.index(target_name)
                    uniq_attrs = set([row[attr_ind] for row in self.data])
                    uniq_targs = set([row[targ_ind] for row in self.data])
                    for k1 in uniq_attrs:
                        for k2 in uniq_targs:
                            if (k1, k2) not in [(row[attr_ind], row[targ_ind]) for row in self.data]:
                                pairs[k1+','+k2] = 0
                            else:
                                for row in self.data:
                                    if row[attr_ind]+','+row[targ_ind] not in pairs.keys():
                                        pairs[row[attr_ind]+','+row[targ_ind]] = 1
                                    elif row[attr_ind]+','+row[targ_ind] in pairs.keys():
                                        pairs[row[attr_ind]+','+row[targ_ind]] += 1
            # print(pairs)
            min_examples = sorted(pairs.items(), key = lambda x:x[1], reverse = False)[0][1]
        #Base case when you've narrowed down the number of people based on splits
        if len(self.data) <= min_examples:
            uniq_targ_vals = set([row[self.feature_names.index(target_name)] for row in self.data])
            max_val_count = 0
            pred_class = None
            for val in uniq_targ_vals:
                val_count = 0
                for row in self.data:
                    if row[self.feature_names.index(target_name)] == val:
                        val_count += 1
                if val_count > max_val_count:
                    pred_class = val
                    max_val_count = val_count
            # print(pred_class+'%: '+str(float(max_val_count*100/len(self.data))))
            return LeafNode(pred_class, float(max_val_count/len(self.data)))
        #Recursive calls
        elif len(self.data) > min_examples:
            #secondary base case in case you run out of columns to split on
            if len(sorted([(self.information_gain(attribute, target_name), attribute) for attribute in self.feature_names if attribute != target_name], key = lambda x: x[0], reverse = True)) <= 1:
                uniq_targ_vals = set([row[self.feature_names.index(target_name)] for row in self.data])
                max_val_count = 0
                pred_class = None
                for val in uniq_targ_vals:
                    val_count = 0
                    for row in self.data:
                        if row[self.feature_names.index(target_name)] == val:
                            val_count += 1
                    if val_count > max_val_count:
                        pred_class = val
                        max_val_count = val_count
                # print(pred_class+'%: '+str(float(max_val_count*100/len(self.data))))
                return LeafNode(pred_class, float(max_val_count/len(self.data)))
            # finding attribute that corresponds to highest info gain and splitting on it
            opt_entropy, split = sorted([(self.information_gain(attribute, target_name), attribute) for attribute in self.feature_names if attribute != target_name], key = lambda x: x[0], reverse = True)[0]
            best_attr = self.feature_names.index(split)
            # print(self.feature_names[best_attr]+' is a split')
            # making a DecisionNode that splits on the attribute chosen from above
            root = DecisionNode(self.feature_names[best_attr], best_attr)
            # figuring out number of splits which is the number of different values that the splitting attribute takes
            branches = set([row[best_attr] for row in self.data])
            # making child DecisionNodes or LeafNodes for each branch
            # print(split+' can take on: '+str(len(branches))+' values')
            for elem in branches:
                data = [] # new table for all the data given attribute = elem, focused data
                for row in self.data:
                    if row[best_attr] == elem: # checking condition in all data
                        r = [thing for thing in row]
                        r.pop(best_attr)
                        data.append(r)
                temp_data = self.data # for restoration after making a child
                # print(len(temp_data)==len(self.data))
                self.data = data # cutting data table for subtree after split
                temp_features = self.feature_names[best_attr] # to reinsert the current splitting attribute into the data table for future children because current child will receive data without this column
                self.feature_names.pop(best_attr) # remove column
                #make a child using DecisionNode.add_child(current splitting attribute, recursive call to DecisionTree.learn() which will always return either a LeafNode or a DecisionNode)
                # print(split+" makes a child node here")
                root.add_child(elem, self.learn(target_name, min_examples))
                self.data = temp_data # restoring full data table for making new child nodes for different values of the current splitting attribute
                self.feature_names.insert(best_attr, temp_features) # need all features for creating next child node since we have to make new branches for current attribute
            self.root = root # setting the splitting node as the DecisionTree's root
            return root

    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        return self.root.children[example.pop(self.root.test_index)].classify(example)

    def __str__(self):
        return self.root.to_str() if self.root else "<empty>"


#############################################

if __name__ == '__main__':
    # print(len(sys.argv))
    path_to_csv = sys.argv[1]
    class_attr_name = sys.argv[2]
    min_examples = int(sys.argv[3])

    # df = pd.read_csv(path_to_csv)
    # tree = DecisionTreeClassifier()
    # X = [df[col] for col in df.columns if col != class_attr_name]
    # Y = df[class_attr_name]
    # tree.fit(X,Y)
    # r = export_text(tree, feature_names = [col for col in df.columns if col != class_attr_name])
    # print(r)

    model = DecisionTree(path_to_csv)
    # print(model.feature_names)
    model.learn(class_attr_name, min_examples)
    # print(model.__str__())
    unreal = ''
    unreal += 'Nay,'*42
    unreal += 'Republican'
    print(model.classify(unreal.split(',')))
    unreal = ''
    unreal += 'Yea,'*42
    unreal += 'Democrat'
    print(model.classify(unreal.split(',')))
    # for i in range(100):
    #   party = ['Democrat','Republican']
    #   choice = ['Yea','Nay','Not Voting']
    #   s = ''
    #   for i in range(42):
    #       s += random.choice(choice)+','
    #   s += random.choice(party)
    #   l = s.split(',')
    #   print(model.classify(l))
    # # print(model.classify('young,short,false,light,brown'.split(',')))