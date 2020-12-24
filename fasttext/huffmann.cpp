#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <cstdio>
#define DEBUG

using namespace std;

class Node {
    public:
    int parent, left, right, weight;
    Node(int parent=-1, int left=-1, int right=-1, int weight=-1): parent(parent), left(left), right(right), weight(weight) { }
    friend ostream& operator<<(ostream& out, const Node& node)
    {
        out << "Node(parent=" << node.parent 
            << ", left=" << node.left << ", right=" 
            << node.right << ", weight=" << node.weight << ")\n";
        return out;
    }
};

vector<Node> buildTree(const vector<int>& weights)
{
    int n = weights.size();
    vector<Node> trees(2 * n - 1);
    for (int i=0; i<n; i++) trees[i].weight = weights[i];
    for (int i=n; i < 2 * n - 1; i++) trees[i].weight = 101;
    sort(trees.begin(), trees.begin() + n, [](const Node& node1, const Node& node2) { 
        return node1.weight > node2.weight;
    });
    int leaf = n-1, node = n;
    for (int i=n; i < 2 * n - 1; i++)
    {
        int smallest[2] = {0, 0};
        for (int j=0; j<2; j++)
        {
            if (leaf >= 0 && trees[leaf].weight < trees[node].weight)
                smallest[j] = leaf--;
            else
                smallest[j] = node++;
        }
        trees[i].left = smallest[0];
        trees[i].right = smallest[1];
        trees[smallest[0]].parent = i;
        trees[i].weight = trees[smallest[0]].weight + trees[smallest[1]].weight;
        trees[smallest[1]].parent = i;
    }
    return trees;
}

int getMinVal(const vector<Node>& trees)
{
    deque<int> q;
    q.push_back(trees.size() - 1);
    int level = -1;
    int ret = 0;
    while(q.size())
    {
        level++;
        int levelsize = q.size();
        for (int i=0; i<levelsize; i++)
        {
            int idx = q.front(); 
            q.pop_front();
            if (trees[idx].left == -1 && trees[idx].right==-1)
                ret += level * trees[idx].weight;
            if (trees[idx].left != -1) {
                q.push_back(trees[idx].left);
            }
            if (trees[idx].right != -1) {
                q.push_back(trees[idx].right);
            }
        }
    }
    return ret;
}

int main()
{
    #ifdef DEBUG
    freopen("text.in", "r", stdin);
    freopen("text.out", "w", stdout);
    #endif

    int n; cin >> n;
    vector<int> weights(n);
    for (int i=0; i<n; i++) cin >> weights[i];
    const vector<Node>& trees = buildTree(weights);
    int ret = getMinVal(trees);
    cout << ret << "\n";

    #ifdef DEBUG
    fclose(stdin);
    fclose(stdout);
    #endif
}