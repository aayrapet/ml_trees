"""
Error handling
"""

# Author: Artur Ayrapetyan


import numpy as np


def check_arguments_data(*args: tuple):
    """
     Universal Error Handler for Data types

    Parameters
    --------
        Args of tuples: first position is the variable, second is the expected data type

    Returns
    --------
        Nothing or error if expected data type is different from actual data type of the given variable

    Examples
    --------

    First Case (no errors, returns nothing)

    >>> a=1
    >>> b=bool
    >>> model=LinearRegression()
    >>> x=np.eye(5)*2
    >>> check_arguments_data((a,int),(b,bool),(model,"__class__"),(x,np.ndarray))

    Second Case (errors, returns value error)

    >>> a=1
    >>> b=bool
    >>> model=LinearRegression()
    >>> x=np.eye(5)*2
    >>> check_arguments_data((a,str),(b,bool),(model,"__class__"),(x,np.ndarray))
    --->       11           pass
               12       else:
    --->       13          raise ValueError (f"{variable} not of type {supposed_type}")
               15 elif supposed_type=="__class__":
               16      if dir(variable)[0]==supposed_type:

                ValueError: 1 not of type <class 'str'>

    """

    for arg in args:
        variable = arg[0]
        supposed_type = arg[1]
        if supposed_type != "__class__":
            if isinstance(variable, supposed_type):
                pass
            else:
                raise ValueError(f"{variable} not of type {supposed_type}")

        elif supposed_type == "__class__":
            if dir(variable)[0] == supposed_type:
                pass
            else:
                raise ValueError(f"{variable} not of type class")


def check_equal_size_vectors(actual, predicted):
    """
    Check if two vectors have equal sizes.

    Parameters:
    actual (array): The actual vector.
    predicted (array): The predicted vector.

    Raises:
    ValueError: If the sizes of the vectors are not equal.
    """
    if len(actual) != len(predicted):
        raise ValueError("arrays must be of equal size")


def check_unique_classes(actual, predicted):
    """
    Check if two vectors have the same unique classes.

    Parameters:
    actual (array): The actual vector containing class labels.
    predicted (array): The predicted vector containing class labels.

    Raises:
    ValueError: If the number of unique classes in the actual vector
                is not equal to the number of unique classes in the predicted vector.
    """
    if len(np.unique(actual)) != len(np.unique(predicted)):
        raise ValueError("not equal classes in two vectors!")


def check_two_classes(actual):
    """
    Check if a vector contains only two unique classes.

    Parameters:
    actual (array): The vector containing class labels.

    Raises:
    ValueError: If the number of unique classes in the vector is not equal to 2,
                indicating multiclass classification, which is out of the scope of this code.
    """
    if len(np.unique(actual)) > 2:
        raise ValueError(
            "for multiclass classification, use weighted calculations, out of the scope of this code"
        )
