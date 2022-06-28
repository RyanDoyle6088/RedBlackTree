"""
Project - Red/Black Trees
Name: Ryan Doyle
"""

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Generator
from Project3.RBnode import RBnode as Node
from copy import deepcopy
import queue

T = TypeVar('T')


class RBtree:
    """
    A Red/Black Tree class
    :root: Root Node of the tree
    :size: Number of Nodes
    """

    __slots__ = ['root', 'size']

    def __init__(self, root: Node = None):
        """ Initializer for an RBtree """
        # this alllows us to initialize by copying an existing tree
        self.root = deepcopy(root)
        if self.root:
            self.root.parent = None
        self.size = 0 if not self.root else self.root.subtree_size()

    def __eq__(self, other: RBtree) -> bool:
        """ Equality Comparator for RBtrees """
        comp = lambda n1, n2: n1isn2 and ((comp(n1.left, n2.left) and comp(n1.right, n2.right)) if (n1 and n2) else True)
        return comp(self.root, other.root) and self.size is other.size

    def __str__(self) -> str:
        """ represents Red/Black tree as string """

        if not self.root:
            return 'Empty RB Tree'

        root, bfs_queue, height= self.root, queue.SimpleQueue(), self.root.subtree_height()
        track = {i:[] for i in range(height+1)}
        bfs_queue.put((root, 0, root.parent))

        while bfs_queue:
            n = bfs_queue.get()
            if n[1] > height:
                break
            track[n[1]].append(n)
            if n[0] is None:
                bfs_queue.put((None, n[1]+1, None))
                bfs_queue.put((None, n[1]+1, None))
                continue
            bfs_queue.put((None, n[1]+1, None) if not n[0].left else (n[0].left, n[1]+1, n[0]))
            bfs_queue.put((None, n[1]+1, None) if not n[0].right else (n[0].right, n[1]+1, n[0]))

        spaces = 12*(2**(height))
        ans = '\n' + '\t\tVisual Level Order Traversal of RBtree'.center(spaces) + '\n\n'
        for i in range(height):
            ans += f"Level {i+1}: "
            for n in track[i]:
                space = int(round(spaces / (2**i)))
                if not n[0]:
                    ans += ' ' * space
                    continue
                ans += "{} ({})".format(n[0], n[2].value if n[2] else None).center(space, " ")
            ans += '\n'
        return ans

    def __repr__(self) -> str:
        return self.__str__()

################################################################
################### Complete Functions Below ###################
################################################################

######################## Static Methods ########################
# These methods are static as they operate only on nodes, without explicitly referencing an RBtree instance

    @staticmethod
    def set_child(parent: Node, child: Node, is_left: bool) -> None:
        """
        Sets the child parameter of parent to child. Which child is determined by the identifier is_left.
        Returns None
        """
        # Node is already created and passed as an argument.
        if is_left:
            if parent.left is None:
                parent.left = child
                child.parent = parent
        else:
            if parent.right is None:
                parent.right = child
                child.parent = parent

    @staticmethod
    def replace_child(parent: Node, current_child: Node, new_child: Node) -> None:
        """
        Replaces parent's child current_child with new_child.
        Returns None
        """
        # Check which side the node is.
        if parent.left is current_child:
            parent.left = new_child
            new_child.parent = parent
        else:
            parent.right = new_child
            new_child.parent = parent

    @staticmethod
    def get_sibling(node: Node) -> Node:
        """
        Given a node, returns the other child of that node's parent, or None should no parent exist.
        """
        sibling = None
        if node.parent is None:
            sibling = None
        else:
            if node.parent.left is not None and node.parent.left != node:    # Sibling is sitting on left.
                sibling = node.parent.left
            if node.parent.right is not None and node.parent.right != node:    # Sibling is sitting on right.
                sibling = node.parent.right
        return sibling

    @staticmethod
    def get_grandparent(node: Node) -> Node:
        """
        Given a node, returns the parent of that node's parent, or None should no such node exist.
        """
        grandparent = None
        if node.parent is not None:
            if node.parent.parent is not None:
                grandparent = node.parent.parent
        return grandparent

    @staticmethod
    def get_uncle(node: Node) -> Node:
        """
        Given a node, returns the sibling of that node's parent, or None should no such node exist.
        """
        uncle = None
        if node.parent is not None:
            if node.parent.parent is not None:
                is_left = None
                if node.parent.parent.right is node.parent:
                    is_left = False
                elif node.parent.parent.left is node.parent:
                    is_left = True
                else:
                    print(' ')
                if is_left:
                    uncle = node.parent.parent.right
                else:
                    uncle = node.parent.parent.left
        return uncle

 ######################## Misc Utilities ##########################

    def min(self, node: Node) -> Node:
        """
        Returns the minimum value stored in the subtree rooted at node, None if the subtree is empty.
        """
        if node is None:
            return None
        while node.left is not None:
            node = node.left
        return node

    def max(self, node: Node) -> Node:
        """
        Returns the maximum value stored in a subtree rooted at node, None if the subtree is empty.
        """
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node

    def search(self, node: Node, val: Generic[T]) -> Node:
        """
        Searches the subtree rooted at node for a node containing value val. If such a node exists,
        return that node, otherwise return the node which would be parent to a node with value val
        should such a node be inserted.
        """
        if node is None:
            return None
        if val is node.value:
            return node
        if val < node.value:
            if node.left is not None:
                return self.search(node.left, val)
            else:
                return node
        if node.right is not None:
            return self.search(node.right, val)
        else:
            return node

 ######################## Tree Traversals #########################

    def inorder(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing an inorder traversal of the subtree rooted at node.
        """
        if node is not None:
            yield from self.inorder(node.left)
            yield node
            yield from self.inorder(node.right)

    def preorder(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing a preorder traversal of the subtree rooted at node.
        """
        if node is not None:
            yield node
            yield from self.preorder(node.left)
            yield from self.preorder(node.right)

    def postorder(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing a postorder traversal of the subtree rooted at node.
        """
        if node is not None:
            yield from self.postorder(node.left)
            yield from self.postorder(node.right)
            yield node

    def bfs(self, node: Node) -> Generator[Node, None, None]:
        """
        Returns a generator object describing a breadth first traversal of the subtree rooted at node.
        """
        h = self.get_tree_height(node)
        for i in range(1, h + 1):
            yield from self.generate_given_level(node, i)   # Print nodes at a given level

    def generate_given_level(self, root, level) -> Generator[Node, None, None]:
        """Helper function for bfs"""
        if root is None:
            return
        if level is 1:
            yield root
        elif level > 1:
            yield from self.generate_given_level(root.left, level-1)
            yield from self.generate_given_level(root.right, level-1)

    def get_tree_height(self, node):
        """Helper function for bfs"""
        if node is None:
            return 0
        else:
            left_height = self.get_tree_height(node.left)
            right_height = self.get_tree_height(node.right)
            if left_height > right_height:
                return left_height+1
            else:
                return right_height+1

################### Rebalancing Utilities ######################

    def left_rotate(self, node: Node) -> None:
        """
        Performs a left tree rotation on the subtree rooted at node.
        """
        nr = node.right
        node.right = nr.left
        if nr.left is not None:
            nr.left.parent = node
        nr.parent = node.parent
        if node.parent is None:
            self.root = nr
        elif node is node.parent.left:
            node.parent.left = nr
        else:
            node.parent.right = nr
        nr.left = node
        node.parent = nr

    def right_rotate(self, node: Node) -> None:
        """
        Performs a right tree rotation on the subtree rooted at node.
        """
        nl = node.left
        node.left = nl.right
        if nl.right is not None:
            nl.right.parent = node
        nl.parent = node.parent
        if node.parent is None:
            self.root = nl
        elif node is node.parent.right:
            node.parent.right = nl
        else:
            node.parent.left = nl
        nl.right = node
        node.parent = nl

    def insertion_repair(self, node: Node) -> None:
        """
        Called after insertion on the node which was inserted, and rebalances the tree by
        ensuring adherance to Red/Black properties.
        """

        while node.parent.is_red:
            if node.parent is node.parent.parent.right:
                uncle = node.parent.parent.left  # uncle
                if uncle is not None and uncle.is_red:
                    uncle.is_red = False
                    node.parent.is_red = False
                    node.parent.parent.is_red = True
                    node = node.parent.parent
                else:
                    if node is node.parent.left:
                        node = node.parent
                        self.right_rotate(node)
                    node.parent.is_red = False
                    node.parent.parent.is_red = True
                    self.left_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.right  # uncle
                if uncle is not None and uncle.is_red:
                    uncle.is_red = False
                    node.parent.is_red = False
                    node.parent.parent.is_red = True
                    node = node.parent.parent
                else:
                    if node is node.parent.right:
                        node = node.parent
                        self.left_rotate(node)
                    node.parent.is_red = False
                    node.parent.parent.is_red = True
                    self.right_rotate(node.parent.parent)
            if node is self.root:
                break
        self.root.is_red = False

    def prepare_removal(self, node: Node) -> None:
        """
        Called prior to removal, on a node that is to be removed.
        It ensures balance is maintained after the removal.
        """
        if self.case_1_rbt(node):
            return
        sibling = self.get_sibling_rbt(node)
        if self.case_2_rbt(node, sibling):
            sibling = self.get_sibling_rbt(node)
        if self.case_3_rbt(node, sibling):
            return
        if self.case_4_rbt(node, sibling):
            return
        if self.case_5_rbt(node, sibling):
            sibling = self.get_sibling_rbt(node)
        if self.case_6_rbt(node, sibling):
            sibling = self.get_sibling_rbt(node)
        sibling.is_red = node.parent.is_red
        node.parent.is_red = False
        if node is node.parent.left:
            sibling.right.is_red = False
            self.left_rotate(node.parent)
        else:
            sibling.left.is_red = False
            self.right_rotate(node.parent)

##################### Insertion and Removal #########################

    def insert(self, node: Node, val: Generic[T]) -> None:
        """
        Inserts an RBnode object to the subtree rooted at node with value val,
        return None.
        """
        node = Node(value=val)
        nxt = None
        slf = self.root
        value_exist = False
        # find node where to insert new node
        while slf is not None:
            nxt = slf
            if node.value is slf.value:
                value_exist = True
                break
            elif node.value < slf.value:
                slf = slf.left
            else:
                slf = slf.right
        # if value already exists don't add
        if value_exist:
            return
        else:
            self.size = self.size + 1
        node.parent = nxt
        if nxt is None:
            self.root = node
        elif node.value < nxt.value:
            nxt.left = node
        else:
            nxt.right = node
        # if new node is a root node no further operation required
        if node.parent is None:
            node.is_red = False
            return
        # if the grandparent is None no further operation required
        if node.parent.parent is None:
            return
        # cross check and fix the tree
        self.insertion_repair(node)

    def remove(self, node: Node, val: Generic[T]) -> None:
        """
        Removes node with value val from the subtree rooted at node. If no such node exists, do nothing.
        """
        node = self.search(self.root, val)
        if node:
            self.remove_node_rbt(node)

    def remove_node_rbt(self, node):
        """Removes node with value val from the subtree rooted at node. If no such node exists, do nothing."""
        if node.left and node.right:
            pred_node = self.get_pred_node_rbt(node)
            pred_node_value = pred_node.value
            self.remove_node_rbt(pred_node)
            node.value = pred_node_value
            return
        if not node.is_red:
            self.prepare_removal(node)
        self.remove_node_by_reference_bst(node)

    @staticmethod
    def get_pred_node_rbt(node):
        """Static method for rbt to get predecessor node, saves time compared
        to doing it in remove function."""
        node = node.left
        while node.right:
            node = node.right
        return node

    @staticmethod
    def case_1_rbt(node):
        """Static method for one of the cases of deletion in rbt,
        just returns true or false if node is red or if parent node is none
        to determine if we can delete node."""
        if node.is_red or node.parent is None:
            return True
        else:
            return False

    def case_2_rbt(self, node, sibling):
        """Method for one of the cases of deletion in rbt,
        if node has red child node, we will replace node with the child
        and return True, else return False and not necessary."""
        if sibling.is_red:
            node.parent.is_red = True
            sibling.is_red = False
            if node is node.parent.left:
                self.left_rotate(node.parent)
            else:
                self.right_rotate(node.parent)
            return True
        return False

    def case_3_rbt(self, node, sibling):
        """Method for one of the cases of deletion in rbt,
        Prepare to remove parent node if met criteria, otherwise
        False and no operation is necessary."""
        if not node.parent.is_red and self.are_both_child_black_rbt(sibling):
            sibling.is_red = True
            self.prepare_removal(node.parent)
            return True
        return False

    def case_4_rbt(self, node, sibling):
        """Method for one of the cases of deletion in rbt,
        if sibling is red, otherwise no operation necessary."""
        if node.parent.is_red and self.are_both_child_black_rbt(sibling):
            node.parent.is_red = False
            sibling.is_red = True
            return True
        return False

    def case_5_rbt(self, node, sibling):
        """Method for one of the cases of deletion in rbt,
        determine if sibling is red, pass to right_rotate and return True,
        if not red no need for operation."""
        if self.is_not_none_and_red_rbt(sibling.left) and self.is_none_or_blank_rbt(sibling.right) \
                and node is node.parent.left:
            sibling.is_red = True
            sibling.left.is_red = False
            self.right_rotate(sibling)
            return True
        return False

    def case_6_rbt(self, node, sibling):
        """Method for one of the cases of deletion in rbt,
        determine if sibling is red, pass to left_rotate and return True,
        if not red no need for operation."""
        if self.is_none_or_blank_rbt(sibling.left) and self.is_not_none_and_red_rbt(sibling.right) \
                and node is node.parent.right:
            sibling.is_red = True
            sibling.right.is_red = False
            self.left_rotate(sibling)
            return True
        return False

    @staticmethod
    def get_sibling_rbt(node):
        """Static method for get sibling node in rbt,
        used for our delete function"""
        if node.parent:
            if node is node.parent.left:
                return node.parent.right
            return node.parent.left
        return None

    @staticmethod
    def are_both_child_black_rbt(node):
        """Static method to see if child node is red, if not
        return True, otherwise False."""
        if node.left and node.left.is_red:
            return False
        if node.right and node.right.is_red:
            return False
        return True

    @staticmethod
    def is_not_none_and_red_rbt(node):
        """Returns false if no node, otherwise recognize node as red."""
        if not node:
            return False
        return node.is_red is True

    @staticmethod
    def is_none_or_blank_rbt(node):
        """Used to see if a node is None, thus not red."""
        if not node:
            return True
        return node.is_red is False

    def remove_node_by_reference_bst(self, node):
        """Removes node in reference to another node, using successor
        or predecessor, returns True if node has been removed or False
        if not."""
        curr_node = node
        # remove curr_node
        if not curr_node.left and not curr_node.right:  # curr_node is leaf
            if not curr_node.parent:
                self.root = None
            elif curr_node.parent.left is curr_node:
                curr_node.parent.left = None
            else:  # curr_node.parent.right is curr_node
                curr_node.parent.right = None
        elif curr_node.left and not curr_node.right:  # curr_node has left only
            if not curr_node.parent:  # curr_node is root
                self.root = curr_node.left
                self.root.is_red = False
                self.root.parent = None
            elif curr_node.parent.left is curr_node:
                curr_node.parent.left = curr_node.left
            else:  # curr_node.parent.left is curr_node
                curr_node.parent.right = curr_node.left
        elif not curr_node.left and curr_node.right:  # curr_node has right only
            if not curr_node.parent:
                self.root = curr_node.right
                self.root.is_red = False
                self.root.parent = None
            elif curr_node.parent.left is curr_node:
                curr_node.parent.left = curr_node.right
            else:  # curr_node = curr_node.parent.right
                curr_node.parent.right = curr_node.right
        else:  # curr_node has left and right children
            successor = curr_node.right
            while successor.left:
                successor_data = successor.value
                self.remove_node_using_bst(successor.value)
                curr_node.value = successor_data
        return True  # node has been removed

    def remove_node_using_bst(self, value):
        """removes the first-found node with the given value, returns True if
         node was found and deleted, parameter of value,
         returns True or False depending on if node was removed."""
        curr_node = self.root
        while curr_node:
            if curr_node.value is value:
                # remove curr_node
                if not curr_node.left and not curr_node.right:  # curr_node is leaf
                    if not curr_node.parent:
                        self.root = None
                    elif curr_node.parent.left is curr_node:
                        curr_node.parent.left = None
                    else:  # in this case curr_node.parent.right is curr_node
                        curr_node.parent.right = None
                elif curr_node.left and not curr_node.right:  # curr_node has left only
                    if not curr_node.parent:  # curr_node is root
                        self.root = curr_node.left
                    elif curr_node.parent.left is curr_node:
                        curr_node.parent.left = curr_node.left
                    else:  # curr_node.parent.left is curr_node
                        curr_node.parent.right = curr_node.left
                elif not curr_node.left and curr_node.right:  # curr_node has right only
                    if not curr_node.parent:
                        self.root = curr_node.right
                    elif curr_node.parent.left is curr_node:
                        curr_node.parent.left = curr_node.right
                    else:  # curr_node = curr_node.parent.right
                        curr_node.parent.right = curr_node.right
                else:  # curr_node has left and right children
                    successor = curr_node.right
                    while successor.left:
                        successor_data = successor.value
                        self.remove_node_using_bst(successor.value)
                        curr_node.value = successor_data
                return True  # node with value has been removed
            elif curr_node.value < value:
                curr_node = curr_node.right
            else:
                curr_node = curr_node.left
        return False
