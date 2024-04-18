import pandas as pd
import numpy as np
from typing import Literal
import _err_handl as erh


class BinaryTreeNode:
    def __init__(
        self,
        index: str,
        condition: np.ndarray,
        next_node: bool,
        entropy: float,
        sums: int,
        unique_v: np.ndarray,
        v_counts: np.ndarray,
        final_node: bool,
        filtered_indices: np.ndarray,
    ):
        """
        Represents a node in a binary tree used for decision tree construction.

        Attributes:
        ------

            index (str): The index of the node .

            condition (np.ndarray): The condition for splitting at this node, represented as an array of booleans.

            next_node (bool): Indicates whether the next node exists or not (if not then this node is a pure node or it is an "artificial" child node of pure node,
                but without splitting condition and "de facto" connection with pure parent node ).

            entropy (float): Entropy value OR Gini value associated with the node.

            sums (int): nb of observations in this node.

            unique_v (np.ndarray): Array of unique classes of this node.

            v_counts (np.ndarray): Array of counts corresponding to unique classes.

            final_node (bool): Indicates whether this is a leaf node/pure node or it is an intermediate node, we will use only (final_node=True nodes) to make a prediction after

            filtered_indices (np.ndarray): In numpy filtering conditions are more trickier then in pandas as in numpy there is no index in  afiltering dataset,
            so we create this array to keep track of indexes of filtered rows of different datasets.

        """
        self.index = index
        self.condition = condition
        self.next_node = next_node
        self.entropy = entropy
        self.left = None
        self.right = None
        self.sums = sums
        self.unique_v = unique_v
        self.v_counts = v_counts
        self.final_node = final_node
        self.filtered_indices = filtered_indices


class DecisionTreeClassifier:
    def __init__(
        self, nb_paths: int, method: Literal["entropy", "gini"], print_mode: bool = True
    ):
        """


        Classic Decision Tree Algorithm used for Classification Tasks.
        ------

        Parameters
        -----

        nb_paths (int): The "depth" of the decision tree that corresponds to number of levels of splits.

        method (str): The formula used for calculating Information Gain:

         - Entropy(S) = - Σ (p_i * log2(p_i))

         - Gini(S) = 1 - Σ (p_i)^2

        print_mode (bool): Print which nodes are leaf/pure (default is True)


        See Also
        ------

        Here is one of the documentations that I find interesting about decision trees:

        https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c

        """
        erh.check_arguments_data((nb_paths, int), (method, str), (print_mode, bool))
        if method not in ["entropy", "gini"]:
            raise ValueError("method has to be entropy or gini")
        self.nb_paths = nb_paths
        self.print_mode = print_mode
        self.method = method

    def formula(self, value_counts: np.ndarray):
        """
        Used to calculate probability of classes defined as number of observations of the class / nb of total observations of the node

        Then , using probability, we calculate Entropy Or Gini index

        """

        # calculate sums
        sums = value_counts.sum()
        if sums != 0:
            # calculate probability
            proba = value_counts / sums
        else:
            proba = np.zeros(len(value_counts))
        # log of 0 does not exist
        proba[proba == 0] = 1

        if self.method == "entropy":
            return (proba @ np.log(proba)) * (-1), sums
        elif self.method == "gini":
            return (1 - (proba @ proba)), sums

    def verif(
        self, unique_values: np.ndarray, value_counts: np.ndarray, on_val_counts: bool
    ):
        """
        When some classes are not found with a split, i want their value counts to be present in the probability vector and set up to 0.

        This is done to make sure that we calculate Gini or entropy index
        """
        if not np.array_equal(np.setdiff1d(self.classes, unique_values), np.array([])):
            not_included = np.setdiff1d(self.classes, unique_values)
            if not on_val_counts:
                unique_values = np.concatenate((not_included, unique_values))
            else:
                value_counts = np.concatenate(
                    (np.zeros(len(not_included)), value_counts)
                )
        return unique_values, value_counts

    @staticmethod
    def comparison_function(
        table: np.ndarray, col: int, val: float, target_index: int, i: int
    ):
        """
        Simple function to split data based on the certain value

        We will have two conditions (for Right Child and Left Child) and two datasets logically

        """

        conditionR = table[:, col] <= val
        conditionL = table[:, col] > val

        splitted_right = table[conditionR][:, target_index]
        splitted_left = table[conditionL][:, target_index]
        return splitted_right, splitted_left, conditionR, conditionL

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit Decision Tree Algorithm


        Parameters
        -----

        x : array_like
              The feature matrix.

        y : array_like
              The target vector.


        """
        erh.check_arguments_data((x, np.ndarray), (y, np.ndarray))

        self.classes = sorted(np.unique(y))
        table = np.column_stack((x, y))

        self.nb_obs = x.shape[0]
        self.target_index = table.shape[1] - 1
        self.variable_index = np.arange(table.shape[1] - 1)

        nodes = self.tree_algorithm(table)
        self.nodes = nodes
        return nodes

    def adj_condition(self, filtered_indices: np.ndarray):
        """
        Adjust numpy condition so that we will have everywhere False , but on the right lines True
        """
        vector = np.zeros(self.nb_obs, dtype=bool)
        vector[filtered_indices] = True
        return vector

    def search_best_split(self, table: np.ndarray, E_parent: float, sums_parent: int):
        """
        This function allows to search through all columns of the dataset and get the threshold with which the split will be made

        For each column we extract unique values of this column and get the "moving average" which will be our threshold .
        For every threshold of this MA we will extract information gain and take the maximum
        """

        start = -float("inf")
        for col_index in self.variable_index:
            unique = np.unique(table[:, col_index])
            unique = self.moving_average(unique)
            for i, val in enumerate(unique):
                # split
                splitted_right, splitted_left, conditionR, conditionL = (
                    self.comparison_function(
                        table, col_index, val, self.target_index, i
                    )
                )
                # count values
                unique_values_r, value_counts_r = np.unique(
                    sorted(splitted_right), return_counts=True
                )
                unique_values_l, value_counts_l = np.unique(
                    sorted(splitted_left), return_counts=True
                )
                # verify value counts
                unique_values_r, value_counts_r = self.verif(
                    unique_values_r, value_counts_r, True
                )
                unique_values_l, value_counts_l = self.verif(
                    unique_values_l, value_counts_l, True
                )
                # calculate entropy/gini
                E_right, sums_r = self.formula(value_counts_r)
                E_left, sums_l = self.formula(value_counts_l)

                # calculate information gain
                IG = (
                    E_parent
                    - (sums_r / sums_parent) * E_right
                    - (sums_l / sums_parent) * E_left
                )
                # maximise entropy /gini
                if IG > start:
                    start = IG
                    this_criterion = val
                    this_column = col_index
                    unique_values_rF = unique_values_r
                    unique_values_lF = unique_values_l
                    value_counts_rF = value_counts_r
                    value_counts_lF = value_counts_l
                    conditionRF, conditionLF = conditionR, conditionL
                    E_r = E_right
                    E_l = E_left
                    su_r = sums_r
                    su_l = sums_l

        return (
            E_r,
            E_l,
            su_r,
            su_l,
            this_column,
            unique_values_rF,
            unique_values_lF,
            value_counts_rF,
            value_counts_lF,
            conditionRF,
            conditionLF,
        )

    @staticmethod
    def moving_average(a: np.ndarray):
        """
        Simple function to calculate moving average of numpy array. Between each value of unique values we will take the mean

        """
        ret = np.cumsum(a, dtype=float)
        ret[2:] = ret[2:] - ret[:-2]
        return ret[2 - 1 :] / 2

    def tree_algorithm(self, table: np.ndarray):
        """
        Decision Tree Algorithm that rassembles all steps above.

        """

        equations = 1
        i = 0
        nodes = {}
        unique_values_parent, value_counts_parent = np.unique(
            sorted(table[:, self.target_index]), return_counts=True
        )
        # define root node ( node 0 )
        E_parent, sums_parent = self.formula(value_counts_parent)

        nodes[f"node_{i}"] = BinaryTreeNode(
            f"node_{i}",  # index (name) of this node
            (True),  # condition that led to this node
            True,  # does next node exist ( child node)
            E_parent,  # entropy
            sums_parent,  # nb obs
            unique_values_parent,  # unique values
            value_counts_parent,  # value counts
            False,  # is final node
            None,  # filtered indices
        )
        ft = True
        max_len_branch=(2**(self.nb_paths-1))*2
        viz_matrix=np.zeros((max_len_branch,self.nb_paths+1))
        h=0
        viz_matrix[0,h]=0
        for k in range(self.nb_paths):
            max_len_branch=2**(self.nb_paths-1-h)
            initial_position_row=0
            h=h+1
            for j in range(equations):
                # definition of nodes
                current_node_name = f"node_{i}"
                right_node_name = f"node_{i+equations+j}"
                left_node_name = f"node_{i+equations+j+1}"
                
                # if parent node has child node
                if nodes[current_node_name].next_node:
                    if ft:

                        filtered_table = table.copy()
                    else:
                        filtered_table = table[nodes[current_node_name].condition]

                    (
                        E_r,
                        E_l,
                        su_r,
                        su_l,
                        this_column,
                        unique_values_rF,
                        unique_values_lF,
                        value_counts_rF,
                        value_counts_lF,
                        conditionRF,
                        conditionLF,
                    ) = self.search_best_split(
                        filtered_table,
                        nodes[current_node_name].entropy,
                        nodes[current_node_name].sums,
                    )
                    if ft:
                        filtered_indicesR = np.where(conditionRF)[0]
                        filtered_indicesL = np.where(conditionLF)[0]
                        ft = False
                    else:
                    
                        filtered_indicesR = nodes[current_node_name].filtered_indices[
                            np.where(conditionRF)[0]
                        ]
                        filtered_indicesL = nodes[current_node_name].filtered_indices[
                            np.where(conditionLF)[0]
                        ]

                    adj_conditionR = self.adj_condition(filtered_indicesR)
                    adj_conditionL = self.adj_condition(filtered_indicesL)
                    for (
                        side_node_name,
                        side_condition,
                        E,
                        su,
                        unique_values,
                        value_counts,
                        filtered_indices_side,
                    ) in [
                        (
                            right_node_name,
                            adj_conditionR,
                            E_r,
                            su_r,
                            unique_values_rF,
                            value_counts_rF,
                            filtered_indicesR,
                        ),
                        (
                            left_node_name,
                            adj_conditionL,
                            E_l,
                            su_l,
                            unique_values_lF,
                            value_counts_lF,
                            filtered_indicesL,
                        ),
                    ]:
                        if len(unique_values) != 1:
                            next_node = True
                        else:
                            next_node = False

                        condition = (side_condition) & (
                            nodes[current_node_name].condition
                        )

                        nodes[side_node_name] = BinaryTreeNode(
                            side_node_name,
                            condition,
                            next_node,
                            E,
                            su,
                            unique_values,
                            value_counts,
                            not next_node,
                            filtered_indices_side,
                        )

                    # connexion of nodes

                    nodes[current_node_name].right = nodes[
                        right_node_name
                    ]  # Right tree

                    nodes[current_node_name].left = nodes[left_node_name]  # left tree
                    
                # if parent node has not child node (pure node or nodes that are dependend on this pure node and which are not defined)
                else:
                    # define empty child nodes and DO NOT connect them to parent pure node
                    nodes[right_node_name] = BinaryTreeNode(
                        right_node_name,
                        "no condition",
                        False,
                        None,
                        None,
                        None,
                        None,
                        False,
                        None,
                    )

                    nodes[left_node_name] = BinaryTreeNode(
                        left_node_name,
                        "no condition",
                        False,
                        None,
                        None,
                        None,
                        None,
                        False,
                        None,
                    )
                    # but if has a condition  then it is a pure node
                    if isinstance((nodes[current_node_name].condition), np.ndarray):
                        if self.print_mode:
                            print("pure node is ", nodes[current_node_name].index)
                
                # if isinstance((nodes[right_node_name].condition), np.ndarray):
                viz_matrix[initial_position_row,h]=i+equations+j
                # else:
                #     viz_matrix[initial_position_row,h]=0
                # if isinstance((nodes[left_node_name].condition), np.ndarray):
                viz_matrix[initial_position_row+max_len_branch,h]=i+equations+j+1
                # else:
                #     viz_matrix[initial_position_row,h]=0
                if (
                    k == (self.nb_paths - 1)
                    and nodes[current_node_name].next_node == True
                ):
                    nodes[current_node_name].right.final_node = True
                    nodes[current_node_name].left.final_node = True
                    if self.print_mode:
                        
                        print("leaf node is : ", nodes[current_node_name].right.index)
                        print("leaf node is : ", nodes[current_node_name].left.index)

                i = i + 1
                initial_position_row=initial_position_row+(max_len_branch)*2

            equations = equations * 2
           
        self.viz=viz_matrix
        return nodes

    def predict(self, x: np.ndarray):
        """
        Predict Y using decision Tree pure/leaf nodes' conditions

        """
        predictions = None
        indexed_table = np.column_stack((np.arange(x.shape[0]), x))
        for el in self.nodes.keys():
            node = self.nodes[el]
            if node.final_node:

                uni_verif, vc_verif = self.verif(node.unique_v, node.v_counts, False)

                index = np.argmax(vc_verif)
                predicted_label = uni_verif[index]
                got = indexed_table[node.condition]
                block_predictions = np.column_stack(
                    (got[:, 0], np.ones(got.shape[0]) * predicted_label)
                )
                if predictions is None:
                    predictions = block_predictions.copy()
                else:
                    predictions = np.row_stack((predictions, block_predictions))

        indices = np.argsort(predictions[:, 0])
        predictions = (predictions[indices])[:, 1]

        return predictions
    
    def initialise(self,nb, indexes):
  
      for i in range(self.nb_paths - 1, -1, -1):
          nb = nb - 1
          if nb >= 0:
              indexes[i] = None

    
    def visualise(self,matrix):
        indexes = {}
        indexes_digits = {}
        loc_indexes_digit = {}
        self.initialise(self.nb_paths - 1, loc_indexes_digit)
        self.initialise(self.nb_paths - 1, indexes_digits)
        old_nz = -float("inf")
        ft = True
        first_line = True
        for line2 in matrix:
            nb_nz = len(np.nonzero(line2)[0])
    
            if nb_nz - old_nz > 1:
                self.initialise(nb_nz, indexes)
            st = ""
            for i in range(len(line2) - 1):
    
                if ft:
                    ft = False
                    st = st + "0"
                    st = st + "----" + str(int(line2[i + 1]))
                else:
                    if (i == indexes[i]) and line2[i] == 0:
                        st = st + "     " + ((indexes_digits[i] - 1) * " ")
                        if line2[i + 1] != 0:
                            st = st + "----" + str(int(line2[i + 1]))
    
                    else:
                        if line2[i] == 0 and line2[i + 1] == 0:
                            st = st + ((indexes_digits[i] - 1) * " ") + "|    "
                        elif line2[i] == 0 and line2[i + 1] != 0:
    
                            st = st + ((indexes_digits[i] - 1) * " ") + "\----"
    
                            index_line = i
                            indexes[index_line] = index_line
                            st = st + str(int(line2[i + 1]))
                        elif line2[i] != 0 and line2[i + 1] != 0:
                            st = st + "----" + str(int(line2[i + 1]))
    
                    if i == indexes[i]:
                        if line2[i] != 0:
                            indexes[i] = None
                if first_line:
                    indexes_digits[i] = len(str(int(line2[i])))
    
                else:
    
                    if line2[i] != 0:
                        loc_indexes_digit[i] = len(str(int(line2[i])))
                    else:
                        loc_indexes_digit[i] = indexes_digits[i]
    
            if loc_indexes_digit != indexes_digits and not first_line:
                indexes_digits = loc_indexes_digit
    
            if first_line:
                first_line = False
            print(st)
            old_nz = nb_nz