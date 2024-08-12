# Ex.No: 2  Implementation of Depth First Search
### DATE: 12-08-2024                                                                         
### REGISTER NUMBER : 212221220025
### AIM: 
To write a python program to implement Depth first Search. 
### Algorithm:
1. Start the program
2. Create the graph by using adjacency list representation
3. Define a function dfs and take the set “visited” is empty 
4. Search start with initial node. Check the node is not visited then print the node.
5. For each neighbor node, recursively invoke the dfs search.
6. Call the dfs function by passing arguments visited, graph and starting node.
7. Stop the program.
### Program:
```
graph = {'A' : ['B','C'],'B' : ['D', 'E'],'D':[],'E':[],'C' : ['F','G'],'F':[],'G':[]}
visited = set() # Set to keep track of visited nodes of graph.
def dfs(visited, graph, node):  #function for dfs 
 if node not in visited:
     print (node)
     visited.add(node)
     for neighbour in graph[node]:
         dfs(visited, graph, neighbour)
# Driver Code
print("Following is the Depth-First Search")
dfs(visited,graph,'A')
```


### Output:
![image](https://github.com/user-attachments/assets/1dbc55c8-ce7d-4572-80a6-9d2c5fae9955)


### Result:
Thus the depth first search order was found sucessfully.
