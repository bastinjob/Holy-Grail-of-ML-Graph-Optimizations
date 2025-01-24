# Tensor DAG Optimization Framework

## Overview

Tensor Graph Optimizer (TGO) is a Python-based project designed to optimize computation graphs for machine learning (ML) and deep learning models. Computation graphs represent ML models as directed acyclic graphs (DAGs), where nodes are operations and edges are tensors. This project identifies and eliminates redundant operations in computation graphs, ensuring efficiency in terms of computation and memory usage.

The optimizer focuses on operations that:

    - Change the values of tensor elements (e.g., element-wise transformations).
    - Change the shape of tensors (e.g., reshaping, transposing).
    - Perform reductions (e.g., summing or averaging elements of tensors).

---


## Key Features
- Rule-based Optimization: Implements a set of optimization rules to simplify redundant or unnecessary operations in computation graphs.
- Support for Various Tensor Operations: Handles element-wise transformations, shape-changing operations, and reductions.
- Efficiency Improvements: Reduces computational overhead and memory usage in ML models.
- Extensible Design: Easily add new optimization rules or tensor operations.


## Optimization Rules

### Candidate Operations on Tensors

Here are the common operations on tensors that can be considered for optimization based on the points you've mentioned:

    1. Element-wise Operations (Value-changing operations):
        Addition (+), Subtraction (-), Multiplication (*), Division (/): These operations change the values of the tensor elements.
        Exponentiation (**), Logarithm (log), Square root (sqrt): Transform tensor values based on specific mathematical functions.
        Activation functions (ReLU, Sigmoid, Tanh, LeakyReLU): Common in neural networks to change tensor values.
        Matrix multiplication: When performing operations like matmul or dot products.

    2. Shape-changing Operations:
        Reshape (reshape, view, expand_dims): Change the shape of a tensor without changing its data.
        Transpose: Swapping axes, typically used to switch the order of dimensions.
        Flatten: Converts multi-dimensional tensors into a 1D tensor.
        Squeeze: Removes dimensions of size 1.
        Concatenate and Stack: Combining tensors along a specified axis.
        Broadcasting: Implicit broadcasting changes the shape of tensors so that they can be operated on together.
        Padding: Adds elements to the tensor along specified dimensions.

    3. Reduction Operations:
        Sum (sum), Mean (mean), Max (max), Min (min): Reduces the tensor along specific axes.
        Standard Deviation (std), Variance (var): Aggregates the tensor's values into a single scalar (or tensor for axis reduction).
        Norm (norm): Computes the norm of a tensor.

### Optimization Rules

Now, we can define a set of rules that govern the optimization of these operations in a computation graph, focusing on reducing redundancy and unnecessary operations:

1. Shape-changing Operations Before Reductions:

    Rule 1: Reshape before reductions is redundant. If a reshape operation occurs immediately before a reduction (e.g., sum, mean), the reshape operation can often be safely removed or merged with the reduction.
        Example: If a tensor is reshaped to (-1, 1) and then a sum is performed, the reshape can be optimized away.
    Rule 2: Transposes before reductions can be optimized. If a transpose operation is followed by a reduction (e.g., sum or mean), the transpose operation can be absorbed into the reduction. For example, transposing rows and columns before taking the sum is unnecessary and can be optimized.

2. Element-wise Operations that Precede Reductions:

    Rule 3: Redundant element-wise operations before reductions: If an element-wise operation is performed on the tensor and the tensor is reduced afterward, and if the operation doesn't change the nature of the tensor in terms of the reduction's outcome (like scaling), the operation can often be eliminated. For instance, performing a sqrt before a sum is unnecessary because the result of the sum will change only the overall magnitude.

    Rule 4: Simplify constant transformations: If an element-wise operation involves a constant (e.g., multiplying by 0 or adding 0), these operations can be simplified. For example:
        tensor + 0 → no-op
        tensor * 1 → no-op
        tensor * 0 → zero tensor

3. Common Subexpression Elimination:

    Rule 5: Avoid duplicate computations: If an operation produces the same intermediate result multiple times, it can be optimized to compute it once and reuse the result.
        Example: If tensor + tensor appears in the graph multiple times, replace it with 2 * tensor.

4. Tensors that Don't Affect the Final Outcome:

    Rule 6: Eliminate unnecessary reshapes or broadcasts before reductions: Any reshape or broadcast that does not impact the final reduction operation can be removed. If the final operation aggregates (e.g., sum, mean), reshapes before the operation don't affect the result.
        Example: Reshaping a tensor for mean operation doesn't change its final output since the mean is computed over the entire tensor regardless of shape.

5. Commutative Operations:

    Rule 7: Reordering commutative operations: Operations like addition and multiplication are commutative, so their order can be optimized for efficiency. For example, moving a large tensor multiplication earlier in the computation might save memory.

6. Reductions over Large Dimensions:

    Rule 8: Optimization of reductions over large dimensions: When reducing a tensor, we can consider whether reducing over a large axis first can reduce memory requirements. For example, performing a sum over rows might be more efficient than reducing over columns depending on memory locality and tensor size.



