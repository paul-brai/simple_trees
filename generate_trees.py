import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        return f"Node({self.value})"


def build_alternating_tree(sequence, depth=3):
    """
    Build a binary tree that represents an alternating sequence.

    Parameters:
    - sequence: The repeating pattern (e.g., [1, 2])
    - depth: Maximum depth of the tree

    Returns:
    - Root node of the constructed tree
    """
    if not sequence:
        return None

    # Create the root with the first value in sequence
    root = TreeNode(sequence[0])

    # Queue for BFS tree construction
    queue = deque([(root, 0)])  # (node, depth)

    while queue:
        node, current_depth = queue.popleft()

        # Stop if we've reached the maximum depth
        if current_depth >= depth:
            continue

        # Calculate values for children based on the sequence pattern
        left_value = sequence[(current_depth + 1) % len(sequence)]
        right_value = sequence[(current_depth + 1) % len(sequence)]

        # Create left child
        node.left = TreeNode(left_value)
        queue.append((node.left, current_depth + 1))

        # Create right child
        node.right = TreeNode(right_value)
        queue.append((node.right, current_depth + 1))

    return root


def build_cyclic_tree(sequence, depth=3):
    """
    Build a binary tree that represents a cyclic sequence.

    Parameters:
    - sequence: The repeating pattern (e.g., [0, 1, 2, 3])
    - depth: Maximum depth of the tree

    Returns:
    - Root node of the constructed tree
    """
    if not sequence:
        return None

    # Create the root with the first value in sequence
    root = TreeNode(sequence[0])

    # Queue for BFS tree construction
    queue = deque([(root, 0)])  # (node, depth)

    while queue:
        node, current_depth = queue.popleft()

        # Stop if we've reached the maximum depth
        if current_depth >= depth:
            continue

        # Calculate values for children based on the sequence pattern and depth
        child_depth = current_depth + 1
        child_value = sequence[child_depth % len(sequence)]

        # Create left child
        node.left = TreeNode(child_value)
        queue.append((node.left, child_depth))

        # Create right child
        node.right = TreeNode(child_value)
        queue.append((node.right, child_depth))

    return root


def serialize_tree(root):
    """
    Convert a tree to a serialized list using pre-order traversal.
    None is used for empty children.
    """
    result = []

    def preorder(node):
        if node is None:
            result.append(None)
            return

        result.append(node.value)
        preorder(node.left)
        preorder(node.right)

    preorder(root)
    return result


def tree_to_tensor(root, max_size=8, pad_value=-1):
    """
    Convert a tree to a fixed-length tensor representation.

    Parameters:
    - root: Root node of the tree
    - max_size: Maximum length of the tensor
    - pad_value: Value to use for padding and None nodes

    Returns:
    - List representation suitable for conversion to PyTorch tensor
    """
    # Serialize the tree
    serialized = serialize_tree(root)

    # Handle size constraints
    if len(serialized) > max_size:
        serialized = serialized[:max_size]  # Truncate
    else:
        # Replace None with pad_value
        serialized = [pad_value if x is None else x for x in serialized]
        # Pad to fixed length
        serialized = serialized + [pad_value] * (max_size - len(serialized))

    return serialized


def visualize_tree(root, level=0, prefix="Root: "):
    """Simple function to print the tree structure"""
    if root is None:
        return

    print(" " * (level * 4) + prefix + str(root.value))
    if root.left or root.right:
        if root.left:
            visualize_tree(root.left, level + 1, "L--- ")
        else:
            print(" " * ((level + 1) * 4) + "L--- None")

        if root.right:
            visualize_tree(root.right, level + 1, "R--- ")
        else:
            print(" " * ((level + 1) * 4) + "R--- None")


def tree_to_networkx(root):
    """
    Convert a binary tree to a NetworkX graph.

    Parameters:
    - root: Root node of the tree

    Returns:
    - A NetworkX DiGraph representing the tree
    """
    if not root:
        return nx.DiGraph()

    G = nx.DiGraph()

    # Use BFS to traverse the tree and add nodes and edges
    queue = deque([(root, None)])  # (node, parent)
    node_counter = 0  # For creating unique node IDs if needed

    while queue:
        current, parent = queue.popleft()

        # Add current node with its value as a node attribute
        if current not in G:
            G.add_node(id(current), value=current.value)

        # Connect parent and current node if parent exists
        if parent:
            G.add_edge(id(parent), id(current))

        # Add children to the queue
        if current.left:
            queue.append((current.left, current))

        if current.right:
            queue.append((current.right, current))

    return G


def visualize_networkx_tree(G, title="Tree Visualization"):
    """
    Visualize a NetworkX graph representing a tree.

    Parameters:
    - G: NetworkX DiGraph
    - title: Title for the plot
    """
    plt.figure(figsize=(12, 8))

    # Use hierarchical layout for tree visualization
    pos = nx.spring_layout(G, seed=42)

    # Get node values for labels
    node_labels = {node: f"{data['value']}" for node, data in G.nodes(data=True)}

    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_color='lightblue', node_size=500,
            font_size=10, font_weight='bold',
            arrows=True, arrowsize=15)

    plt.title(title)
    plt.axis('off')
    # plt.tight_layout()
    plt.show()


def build_repeating_tree(sequence, depth=3):
    """
    Build a binary tree that represents a repeating sequence of any length.

    Parameters:
    - sequence: The repeating pattern (e.g., [1, 2, 3])
    - depth: Maximum depth of the tree

    Returns:
    - Root node of the constructed tree
    """
    if not sequence:
        return None

    # Create the root with the first value in sequence
    root = TreeNode(sequence[0])

    # Queue for BFS tree construction
    queue = deque([(root, 0)])  # (node, depth)

    while queue:
        node, current_depth = queue.popleft()

        # Stop if we've reached the maximum depth
        if current_depth >= depth:
            continue

        # Calculate values for children based on the sequence pattern
        next_depth = current_depth + 1
        next_value = sequence[next_depth % len(sequence)]

        # Create left child
        node.left = TreeNode(next_value)
        queue.append((node.left, next_depth))

        # Create right child
        node.right = TreeNode(next_value)
        queue.append((node.right, next_depth))

    return root

def extract_values_dfs(root):
    """Extract values via DFS pre-order traversal"""
    values = []

    def dfs(node):
        if node is None:
            return
        values.append(node.value)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return values


# Example usage
if __name__ == "__main__":
    # Alternating sequence [1, 2, 1, 2, ...]
    alt_sequence = [1, 2, 3]
    alt_tree = build_repeating_tree(alt_sequence, depth=4)

    # Extract values to verify pattern
    values = extract_values_dfs(alt_tree)
    print("\nValues from DFS traversal:", values[:15])
    #
    # # Convert to NetworkX
    # alt_graph = tree_to_networkx(alt_tree)
    # print(f"Alternating Tree Graph: {alt_graph.number_of_nodes()} nodes, {alt_graph.number_of_edges()} edges")
    #
    # # Print node values
    # print("Node values:")
    # for node, data in alt_graph.nodes(data=True):
    #     print(f"Node {node}: value = {data['value']}")
    #
    # print("Tensor representation: ", tree_to_tensor(alt_tree))
    #
    # # Visualize
    # visualize_networkx_tree(alt_graph, "Alternating Sequence Tree")
    #
    # # Cyclic sequence [0, 1, 2, 3, 0, 1, 2, 3, ...]
    # cyclic_sequence = [0, 1, 2, 3]
    # cyclic_tree = build_cyclic_tree(cyclic_sequence, depth=2)
    #
    # # Convert to NetworkX
    # cyclic_graph = tree_to_networkx(cyclic_tree)
    # print(f"\nCyclic Tree Graph: {cyclic_graph.number_of_nodes()} nodes, {cyclic_graph.number_of_edges()} edges")
    # #
    # # Visualize
    # visualize_networkx_tree(cyclic_graph, "Cyclic Sequence Tree")
    #
    # visualize_tree(cyclic_tree)
    #
    # print("Tensor representation: ", tree_to_tensor(cyclic_tree))

    # # You can now use these NetworkX graphs for various graph algorithms and analyses
    # # For example:
    # print("\nTree properties:")
    # print(f"Is connected: {nx.is_connected(alt_graph.to_undirected())}")
    # print(f"Diameter: {nx.diameter(alt_graph.to_undirected())}")
    # print(f"Average shortest path length: {nx.average_shortest_path_length(alt_graph)}")