from copy import deepcopy

class GraphUtil:

    def CausesCycle(self,adjmat,child,parent):
        ag = deepcopy(adjmat);
        ag = ag.T;
        V  = ag.shape[0];
        temp_graph=[];
        for i in range(V):
            rw=[];
            for j in range(V):
                rw.append(ag[i][j])
            temp_graph.append(rw);
        temp_graph[child][parent]=1;
        flag= self.HasCycle(temp_graph);
        return flag;

    def HasCycle(self,graph):
        V = len(graph);
        visited = [False] * V;
        onStack = [False] * V;

        for n in range(V):
            if not visited[n]:
                if self.CycleChecker(graph,n,visited,onStack) == True:
                    return True;
        return False;

    def CycleChecker(self,graph,node,visited,stack):
        visited[node]=True;
        stack[node]=True;

        neighbours=[i for i,x in enumerate(graph[node]) if x !=0];

        for neighbour in neighbours:
            if not visited[neighbour]:
                if self.CycleChecker(graph,neighbour,visited,stack):
                    return True;
            elif stack[neighbour]==True:
                return True;

        stack[node]=False;
        return False;

