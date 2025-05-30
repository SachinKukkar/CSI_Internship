class Node:
    """Represents a node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Manages a singly linked list."""
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Adds a node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            print(f"Added head node: {data}")
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            print(f"Added node: {data}")

    def print_list(self):
        """Prints the data in the list."""
        current = self.head
        if not current:
            print("The list is empty.")
            return
        print("Linked List:", end=" ")
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """Deletes the nth node (1-based index) from the list."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")

        if n <= 0:
            raise IndexError("Index must be a positive integer.")

        # Deleting the head node
        if n == 1:
            print(f"Deleting node at position {n}: {self.head.data}")
            self.head = self.head.next
            return

        # Traverse to the (n-1)th node
        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1

        if not current or not current.next:
            raise IndexError("Index out of range.")

        print(f"Deleting node at position {n}: {current.next.data}")
        current.next = current.next.next


# Sample usage
if __name__ == "__main__":
    ll = LinkedList()
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    ll.print_list()

    try:
        ll.delete_nth_node(2)
        ll.print_list()

        ll.delete_nth_node(1)
        ll.print_list()

        ll.delete_nth_node(10)  # Should raise IndexError
    except Exception as e:
        print("Error:", e)
