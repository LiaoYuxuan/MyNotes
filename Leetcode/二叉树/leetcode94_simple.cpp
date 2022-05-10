
// 递归方法
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        //返回空数组的方法
        if(root==nullptr) return {};

        vector<int> l,r,inorder;
        int m;
        l=inorderTraversal(root->left);
        m=root->val;
        r=inorderTraversal(root->right);
        inorder.insert(inorder.end(), l.begin(), l.end());
        inorder.push_back(m);
        inorder.insert(inorder.end(), r.begin(), r.end());
        return inorder;
    }
};


// 迭代方法（即用到栈）:各种遍历通用
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        //返回空数组的方法
        if(root==nullptr) return {};
        //栈用于存放树节点（类型易写错）
        stack<TreeNode*> s;
        TreeNode* p = root;
        vector<int> inorder;
        //外循环从左子节点入手(注意逻辑运算符是“或”)
        while(p!=nullptr || s.size()!=0){
            s.push(p);
            p = p->left;
            //内循环从右子节点入手
            while(p==nullptr && s.size()!=0){
                p = s.top();
                inorder.push_back(p->val);
                s.pop();
                p = p->right;
            }
        }
        return inorder;
    }
};

// 颜色遍历法（前一种迭代方法的部分改进，主要好理解一些，容易扩展）
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        //返回空数组的方法
        if(root==nullptr) return {};
        //栈用于存放树节点（类型易写错）
        stack<TreeNode*> s;
        stack<int> flag;
        s.push(root);
        flag.push(0);
        TreeNode* p;
        int color;
        vector<int> inorder;
        while(s.size()!=0){
            p = s.top();
            color = flag.top();
            s.pop();
            flag.pop();
            if(p==nullptr) continue;
            // 0对应白色 1对应灰色
            if(color==0){
                //中序遍历：按照右、左、中的顺序入栈
                s.push(p->right);
                flag.push(0);
                s.push(p);
                flag.push(1);
                s.push(p->left);
                flag.push(0);
            }else{
                inorder.push_back(p->val);
            }
        }
        return inorder;
    }
};