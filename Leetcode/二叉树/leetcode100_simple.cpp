class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        //分为三步：根节点的值是否相同，左子树的根结点值是否相同，右子树的根结点值是否相同

        //对于根结点为空进行判断
        if(p==nullptr && q==nullptr) return true;
        else if(p==nullptr || q==nullptr) return false;

        //跟节点不为空，则看值是否一样
        if(p->val!=q->val) return false;
        else{
            //值一样，则递归访问左子树和右子树
            return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        }

    }
};