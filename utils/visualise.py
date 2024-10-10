import treelib
import graphviz


def save_tree(node, title):
    tree = treelib.Tree()

    def add_nodes_to_tree(node, tree, parent_id=None):
        node_tag = f"e: {node.epoch}, n: {node.n}, q: {node.q}, core_acc: {node.core_acc}, core_loss: {node.core_loss}"
        # Include n and q values in the node tag
        current_id = tree.create_node(tag=node_tag, data=node, parent=parent_id)
        for child in node.children:
            add_nodes_to_tree(child, tree, parent_id=current_id)

    add_nodes_to_tree(node, tree)

    tree.save2file(f"{title}_structure.txt")
    tree.to_graphviz(f"{title}_structure_graphviz.dot")


def tree_to_graphviz():
    dot = graphviz.Source.from_file('all_structure_graphviz.dot')
    # dot.view()


tree_to_graphviz()